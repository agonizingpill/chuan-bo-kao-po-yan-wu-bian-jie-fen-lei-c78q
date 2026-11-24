"""Frozen-observation training samples for tide-window overtime prediction.

A sample is only produced for a case that received nothing but tracking
instructions (``Action.MONITOR``) before close.  The label compares the
recorded all-fast time against the tide window that the berth request was
registered under; the window end is frozen onto the sample so a later
correction cannot silently rewrite the credential.

Lifecycle::

    QUEUED --worker--> GENERATED --supervisor--> PUBLISHED
       \\
        `--worker error--> FAILED --retry (same row)--> QUEUED

Samples are versioned.  A correction that lands while the newest version is
not yet published rewrites that pending version and re-queues it, so the
publication that follows uses the corrected label.  A correction that lands
after a version was published starts a new version and leaves the published
one with its original label untouched.
"""

import logging

from redis.exceptions import RedisError
from sqlalchemy import select
from sqlalchemy.orm import Session as DatabaseSession

from .classifier import FEATURES, snapshot_digest
from .database import WAKE_QUEUE, queue
from .models import (
    Action,
    BerthRequest,
    DelayIncident,
    DispatchReview,
    IncidentState,
    SampleLabel,
    SampleState,
    TideWindow,
    TrainingSample,
)
from .planning import Rejected, audit, find, utc_now

log = logging.getLogger(__name__)


def notify():
    try:
        queue.lpush(WAKE_QUEUE, 'training-sample')
    except RedisError:
        log.exception('Redis wake-up failed for training sample; worker scans durable queue')


def label_for(all_fast_at, window_ends_at):
    return SampleLabel.OVERTIME if all_fast_at > window_ends_at else SampleLabel.ON_TIME


def tracking_only(db: DatabaseSession, incident_code):
    actions = list(db.scalars(select(DispatchReview.action).where(DispatchReview.incident_code == incident_code)))
    return bool(actions) and all(action == Action.MONITOR for action in actions)


def _window(db, incident):
    request = find(db, BerthRequest, incident.request_code)
    return find(db, TideWindow, request.window_code)


def latest(db: DatabaseSession, incident_code, *, lock=False):
    statement = (select(TrainingSample).where(TrainingSample.incident_code == incident_code)
                 .order_by(TrainingSample.version.desc()).limit(1))
    if lock:
        statement = statement.with_for_update()
    return db.scalar(statement)


def _fill(sample, incident, record, window):
    sample.observed_features = dict(incident.features)
    sample.input_digest = incident.input_digest
    sample.window_code = window.code
    sample.window_ends_at = window.ends_at
    sample.all_fast_at = record.all_fast_at
    sample.label = label_for(record.all_fast_at, window.ends_at)


def _new_version(db, actor, incident, record, version):
    window = _window(db, incident)
    sample = TrainingSample(
        incident_code=incident.code,
        version=version,
        state=SampleState.QUEUED,
        generated_by=actor,
        generated_at=utc_now(),
    )
    _fill(sample, incident, record, window)
    db.add(sample)
    db.flush()
    audit(db, actor, incident.code, 'SAMPLE_QUEUED',
          f'v{sample.version} · {sample.label} · 潮窗 {window.code} 止 {window.ends_at:%m-%d %H:%M}')
    return sample


def create_for_close(db: DatabaseSession, actor, incident, record):
    """Queue the first sample version when a tracking-only case is closed."""
    if record.all_fast_at is None or not tracking_only(db, incident.code):
        return None
    existing = latest(db, incident.code)
    if existing:
        return existing
    return _new_version(db, actor, incident, record, 1)


def correct(db: DatabaseSession, actor, code, data):
    """Supervisor correction of the all-fast credential on a closed case."""
    incident = find(db, DelayIncident, code, lock=True)
    if incident.state != IncidentState.CLOSED:
        raise Rejected('CORRECTION_STATE', '只有已结案案件可以补正靠妥时间', code)
    record = find(db, DispatchReview, code, lock=True)
    if data.all_fast_at < incident.observed_at:
        raise Rejected('ALL_FAST_TIME', '全缆系妥时刻不得早于延误观测', code)
    previous = record.all_fast_at
    record.all_fast_at = data.all_fast_at
    audit(db, actor, code, 'ALL_FAST_CORRECTED',
          f'{previous:%m-%d %H:%M} → {data.all_fast_at:%m-%d %H:%M} · {data.reason}')
    if not tracking_only(db, code):
        return record
    pending = latest(db, code, lock=True)
    if pending and pending.state != SampleState.PUBLISHED:
        # The correction landed before publication: re-run this pending version
        # with the corrected credential instead of adding a duplicate sample.
        # Keep the window frozen at registration; only the credential moves.
        pending.all_fast_at = record.all_fast_at
        pending.label = label_for(record.all_fast_at, pending.window_ends_at)
        pending.state = SampleState.QUEUED
        pending.error = None
        pending.published_at = None
        pending.generated_by = actor
        pending.generated_at = utc_now()
        audit(db, actor, code, 'SAMPLE_REQUEUED',
              f'v{pending.version} · 补正后重跑 · {pending.label}')
    else:
        version = (pending.version if pending else 0) + 1
        _new_version(db, actor, incident, record, version)
    return record


def publish(db: DatabaseSession, actor, sample_id):
    """Admit a generated sample into the training corpus."""
    sample = find(db, TrainingSample, sample_id, lock=True)
    if sample.state == SampleState.PUBLISHED:
        return sample
    if sample.state != SampleState.GENERATED:
        raise Rejected('SAMPLE_PUBLISH_STATE', '只有已生成且未发布的样本可以发布', sample.incident_code)
    sample.state = SampleState.PUBLISHED
    sample.published_at = utc_now()
    audit(db, actor, sample.incident_code, 'SAMPLE_PUBLISHED',
          f'v{sample.version} · {sample.label} · 靠妥 {sample.all_fast_at:%m-%d %H:%M}')
    return sample


def retry(db: DatabaseSession, actor, sample_id):
    sample = find(db, TrainingSample, sample_id, lock=True)
    if sample.state != SampleState.FAILED:
        raise Rejected('SAMPLE_RETRY_STATE', '只有生成失败的样本可以重试', sample.incident_code)
    sample.state = SampleState.QUEUED
    sample.error = None
    audit(db, actor, sample.incident_code, 'SAMPLE_REQUEUED',
          f'v{sample.version} · 重试生成 · 沿用同一入样记录')
    return sample


def generate(db: DatabaseSession, sample):
    """Worker step: validate the frozen observation and derive the final label."""
    incident = find(db, DelayIncident, sample.incident_code)
    features = sample.observed_features or {}
    if set(features) != set(FEATURES):
        raise ValueError('冻结观测字段不完整')
    if snapshot_digest(features) != sample.input_digest:
        raise ValueError('冻结观测摘要不一致')
    if sample.all_fast_at < incident.observed_at:
        raise ValueError('靠妥时间早于延误观测')
    sample.label = label_for(sample.all_fast_at, sample.window_ends_at)
    sample.state = SampleState.GENERATED
    sample.error = None
    audit(db, 'worker', sample.incident_code, 'SAMPLE_GENERATED',
          f'v{sample.version} · {sample.label} · 靠妥 {sample.all_fast_at:%m-%d %H:%M}')
    return sample
