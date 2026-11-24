"""Model-level tests for the frozen-observation training samples.

These run against an in-memory SQLite database so the sample lifecycle,
correction-vs-publication ordering, and retry-idempotency can be checked
without MySQL or Redis.
"""

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from src import samples
from src.models import (
    Action,
    Base,
    Berth,
    BerthRequest,
    DelayIncident,
    DispatchReview,
    IncidentState,
    RequestState,
    Role,
    SampleLabel,
    SampleState,
    TideWindow,
    TrainingSample,
    Tug,
    User,
)
from src.schemas import CorrectionInput


OBSERVED = datetime(2026, 11, 21, 9, 48)
WINDOW_START = datetime(2026, 11, 21, 9, 10)
WINDOW_END = datetime(2026, 11, 21, 10, 45)
FEATURES = {'eta_offset_min': 18.0, 'tide_reserve_min': 25.0, 'wind_kn': 14.0, 'tug_wait_min': 12.0}


@pytest.fixture()
def db():
    engine = create_engine('sqlite://', connect_args={'check_same_thread': False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    with Session(engine, expire_on_commit=False) as session:
        session.add_all([
            User(username='supervisor', role=Role.SUPERVISOR, password_hash='x'),
            Berth(code='B07', label='东七码头'),
            TideWindow(code='TW-072', berth_code='B07', starts_at=WINDOW_START, ends_at=WINDOW_END,
                       max_draft_m=13.2, bulletin='TIDE-EAST-1121-AM.pdf'),
            BerthRequest(code='BR-413', vessel='远桥 75', berth_code='B07', window_code='TW-072',
                         eta=datetime(2026, 11, 21, 9, 30), maneuver_minutes=50, draft_m=11.9,
                         state=RequestState.COMPLETED),
        ])
        session.commit()
        yield session


def make_incident(db, code='DL-413'):
    from src.classifier import snapshot_digest
    incident = DelayIncident(
        code=code, request_code='BR-413', radio_reference='VHF-12/0930-19', observed_at=OBSERVED,
        event_note='引航通报拖轮等候', state=IncidentState.CLOSED, features=dict(FEATURES),
        input_digest=snapshot_digest(FEATURES),
    )
    db.add(incident)
    db.flush()
    return incident


def make_review(db, code='DL-413', action=Action.MONITOR, all_fast_at=None):
    record = DispatchReview(
        incident_code=code, action=action, instruction='保持跟踪并记录实绩', issued_by='supervisor',
        issued_at=OBSERVED + timedelta(minutes=5),
        all_fast_at=all_fast_at or datetime(2026, 11, 21, 10, 50),
        outcome='作业完成', closed_by='supervisor', closed_at=datetime(2026, 11, 21, 10, 55),
    )
    db.add(record)
    db.flush()
    return record


def close_case(db, incident, record):
    sample = samples.create_for_close(db, 'supervisor', incident, record)
    db.flush()
    return sample


def test_close_tracking_case_creates_overtime_sample(db):
    incident = make_incident(db)
    record = make_review(db, all_fast_at=datetime(2026, 11, 21, 10, 50))
    sample = close_case(db, incident, record)
    assert sample is not None
    assert sample.version == 1
    assert sample.label == SampleLabel.OVERTIME
    assert sample.window_ends_at == WINDOW_END
    assert sample.state == SampleState.QUEUED
    assert sample.observed_features == FEATURES


def test_close_on_time_when_all_fast_before_window_end(db):
    incident = make_incident(db)
    record = make_review(db, all_fast_at=datetime(2026, 11, 21, 10, 30))
    sample = close_case(db, incident, record)
    assert sample.label == SampleLabel.ON_TIME


def test_non_tracking_case_is_not_sampled(db):
    incident = make_incident(db)
    record = make_review(db, action=Action.STANDBY)
    sample = close_case(db, incident, record)
    assert sample is None
    assert db.scalar(select(TrainingSample)) is None


def test_correction_before_publication_requeues_same_version(db):
    incident = make_incident(db)
    record = make_review(db, all_fast_at=datetime(2026, 11, 21, 10, 50))
    original = close_case(db, incident, record)
    assert original.label == SampleLabel.OVERTIME

    samples.correct(db, 'supervisor', 'DL-413', CorrectionInput(
        all_fast_at=datetime(2026, 11, 21, 10, 30), reason='引航席补录系缆完成时间'))
    db.flush()

    rows = list(db.scalars(select(TrainingSample).order_by(TrainingSample.version)))
    assert len(rows) == 1
    assert rows[0].version == 1
    assert rows[0].label == SampleLabel.ON_TIME
    assert rows[0].state == SampleState.QUEUED
    assert rows[0].all_fast_at == datetime(2026, 11, 21, 10, 30)


def test_correction_after_publication_starts_new_version_and_keeps_old_label(db):
    incident = make_incident(db)
    record = make_review(db, all_fast_at=datetime(2026, 11, 21, 10, 50))
    first = close_case(db, incident, record)
    samples.generate(db, first)
    samples.publish(db, 'supervisor', first.id)
    db.flush()
    assert first.state == SampleState.PUBLISHED
    assert first.label == SampleLabel.OVERTIME

    samples.correct(db, 'supervisor', 'DL-413', CorrectionInput(
        all_fast_at=datetime(2026, 11, 21, 10, 30), reason='核对靠泊记录后补正'))
    db.flush()

    rows = list(db.scalars(select(TrainingSample).order_by(TrainingSample.version)))
    assert [row.version for row in rows] == [1, 2]
    assert rows[0].state == SampleState.PUBLISHED
    assert rows[0].label == SampleLabel.OVERTIME  # untouched
    assert rows[1].version == 2
    assert rows[1].label == SampleLabel.ON_TIME
    assert rows[1].state == SampleState.QUEUED


def test_retry_failed_sample_reuses_row_without_duplication(db):
    incident = make_incident(db)
    record = make_review(db)
    sample = close_case(db, incident, record)
    sample.state = SampleState.FAILED
    sample.error = '冻结观测摘要不一致'
    db.flush()

    samples.retry(db, 'supervisor', sample.id)
    db.flush()
    rows = list(db.scalars(select(TrainingSample)))
    assert len(rows) == 1
    assert rows[0].id == sample.id
    assert rows[0].state == SampleState.QUEUED
    assert rows[0].error is None


def test_correction_rejected_outside_closed_state(db):
    incident = make_incident(db)
    record = make_review(db)
    close_case(db, incident, record)
    incident.state = IncidentState.ACTIONED
    db.flush()
    from src.planning import Rejected
    with pytest.raises(Rejected):
        samples.correct(db, 'supervisor', 'DL-413', CorrectionInput(
            all_fast_at=datetime(2026, 11, 21, 10, 30), reason='测试补正依据'))
