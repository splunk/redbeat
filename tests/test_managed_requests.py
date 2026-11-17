from datetime import timedelta
from unittest.mock import patch

import pytest
from celery.schedules import schedule
from redis.exceptions import WatchError

from redbeat.schedulers import RedBeatScheduler, RedBeatSchedulerEntry
from tests.test_jitter_concurrent_updates import app as request_app
from tests.test_jitter_concurrent_updates import create_entry, load
from tests.test_jitter_integration import NOW
from tests.test_jitter_integration import app as redis_app  # noqa: F401

app = request_app


def test_delayed_request_uses_existing_anchor_and_guards_lookahead(app):
    entry = create_entry(app, NOW, kwargs={'value': 'old'})
    next(entry)
    entry = load(entry)
    due = NOW + timedelta(seconds=80)
    entry.kwargs = {'value': 'new'}
    entry.request_run(due)
    stored = load(entry)
    assert stored.kwargs == {'value': 'new'}
    assert stored.last_run_at == due - timedelta(seconds=60)
    assert stored.total_run_count == 1
    assert app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, entry.key) == due.timestamp()
    assert stored._strtobool(stored.run_immediately)
    assert not app.redbeat_redis.sismember('run_immediately', entry.key)
    scheduler = RedBeatScheduler(app, lazy=True, max_interval=3600)
    try:
        with patch.object(app, 'send_task') as publish:
            for seconds in (0, 60, 79.999999):
                app.now = lambda: NOW + timedelta(seconds=seconds)
                scheduler.tick()
                publish.assert_not_called()
            app.now = lambda: due
            scheduler.tick()
            publish.assert_called_once()
        finished = load(entry)
        assert finished.last_run_at == due
        assert finished.total_run_count == 2
        assert not finished._strtobool(finished.run_immediately)
        assert (
            app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, entry.key)
            == (due + timedelta(seconds=60)).timestamp()
        )
    finally:
        scheduler.close()


@pytest.mark.parametrize('order', [(False, True), (True, False)])
def test_immediate_request_wins_in_either_arrival_order(app, order):
    entry = create_entry(app, first_run_kwargs={'bootstrap': 'first'})
    for index, run_immediately in enumerate(order):
        current = load(entry)
        current.kwargs = {'revision': index}
        due = NOW + timedelta(seconds=0 if run_immediately else 20)
        current.request_run(due, run_immediately=run_immediately)
    stored = load(entry)
    assert stored.kwargs == {'revision': 1}
    assert stored.last_run_at == NOW - timedelta(seconds=60)
    assert app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, entry.key) == NOW.timestamp()
    assert stored.first_run_kwargs == {}
    assert stored._strtobool(stored.run_immediately)
    assert app.redbeat_redis.sismember('run_immediately', entry.key)
    assert stored.is_due()[0]
    outgoing = next(stored)
    assert 'bootstrap' not in outgoing.kwargs
    assert not outgoing._strtobool(outgoing.run_immediately)
    finished = load(entry)
    assert not finished._strtobool(finished.run_immediately)
    assert not app.redbeat_redis.sismember('run_immediately', entry.key)


def test_repeated_deferred_requests_keep_earliest_due_and_latest_payload(app):
    entry = create_entry(app)
    for index, seconds in enumerate((20, 40, 10)):
        current = load(entry)
        current.kwargs = {'revision': index}
        current.request_run(NOW + timedelta(seconds=seconds))
        accepted_due = NOW + timedelta(seconds=min((20, 40, 10)[: index + 1]))
        assert (
            app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, entry.key)
            == accepted_due.timestamp()
        )
        assert load(entry).last_run_at == accepted_due - timedelta(seconds=60)
        assert load(entry)._strtobool(load(entry).run_immediately)
    assert load(entry).kwargs == {'revision': 2}
    assert set(app.redbeat_redis.hgetall(entry.key)) == {
        'definition',
        'meta',
        'jitter_fingerprint',
        'entry_id',
    }


@pytest.mark.parametrize('new_interval', [60, 120])
@pytest.mark.parametrize('request_seconds', [10, 40])
def test_refresh_preserves_pending_first_arguments_and_earliest_deadline(
    app, new_interval, request_seconds
):
    entry = create_entry(app, NOW + timedelta(seconds=20), first_run_kwargs={'bootstrap': 'first'})
    entry.kwargs = {'revision': 'latest'}
    entry.schedule = schedule(new_interval, app=app)
    entry.request_run(
        NOW + timedelta(seconds=request_seconds),
        fingerprint=entry.jitter_fingerprint,
        preserve_first_run_kwargs=True,
    )
    stored = load(entry)
    due = NOW + timedelta(seconds=min(20, request_seconds))
    assert stored.first_run_kwargs == {'bootstrap': 'first'}
    assert stored.kwargs == {'revision': 'latest'}
    assert stored.last_run_at + stored.schedule.run_every == due
    assert stored.total_run_count == 0
    app.now = lambda: due
    outgoing = next(stored)
    assert outgoing.kwargs == {'revision': 'latest', 'bootstrap': 'first'}
    assert load(entry).first_run_kwargs == {}
    assert load(entry).total_run_count == 1


def test_refresh_does_not_restore_already_consumed_first_arguments(app):
    entry = create_entry(app, first_run_kwargs={'bootstrap': 'first'})
    stale = load(entry)
    next(entry)
    stale.request_run(NOW + timedelta(seconds=20), preserve_first_run_kwargs=True)
    stored = load(entry)
    assert stored.first_run_kwargs == {}
    assert stored.total_run_count == 1
    app.now = lambda: NOW + timedelta(seconds=20)
    assert 'bootstrap' not in next(stored).kwargs


@pytest.mark.parametrize('writer', ['save', 'reschedule'])
@pytest.mark.parametrize('loaded', [False, True])
def test_legacy_immediate_write_does_not_inherit_a_managed_deadline(app, writer, loaded):
    original = create_entry(app, NOW + timedelta(seconds=20))
    if loaded:
        entry = load(original)
    else:
        entry = RedBeatSchedulerEntry(original.name, original.task, schedule(60, app=app), app=app)
    entry.run_immediately = True
    getattr(entry, writer)()
    stored = load(original)
    assert entry.jitter_fingerprint is None
    assert stored.jitter_fingerprint is None
    assert stored.entry_id == original.entry_id
    assert stored.is_due()[0]
    next(stored)
    assert load(original).last_run_at == NOW
    assert load(original).total_run_count == 1


def test_definition_update_preserves_complete_pending_state(app):
    entry = create_entry(app)
    entry.request_run(NOW + timedelta(seconds=20))
    before = app.redbeat_redis.hgetall(entry.key)
    score = app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, entry.key)
    entry.kwargs = {'value': 'updated'}
    entry.run_immediately = True
    entry.update_definition()
    after = app.redbeat_redis.hgetall(entry.key)
    assert after['meta'] == before['meta']
    assert after['jitter_fingerprint'] == before['jitter_fingerprint']
    assert app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, entry.key) == score
    assert load(entry)._strtobool(load(entry).run_immediately)


@pytest.mark.parametrize('run_immediately', [False, True])
@pytest.mark.parametrize('loaded_request', [False, True])
def test_new_request_survives_older_reservation(app, run_immediately, loaded_request):
    entry = create_entry(app)
    if loaded_request:
        entry.request_run(NOW)
    stale = load(entry)
    current = load(entry)
    current.kwargs = {'value': 'new'}
    due = NOW if run_immediately else NOW + timedelta(seconds=20)
    current.request_run(due, run_immediately=run_immediately)
    before = app.redbeat_redis.hgetall(entry.key)
    score = app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, entry.key)
    with pytest.raises(ValueError):
        next(stale)
    after = app.redbeat_redis.hgetall(entry.key)
    assert after == before
    assert app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, entry.key) == score
    assert load(entry).total_run_count == 0
    app.now = lambda: NOW + timedelta(seconds=20)
    assert load(entry).is_due()[0]
    next(load(entry))
    assert not load(entry)._strtobool(load(entry).run_immediately)
    assert load(entry).total_run_count == 1


def test_identical_requested_runs_can_coalesce_without_additional_state(app):
    entry = create_entry(app, kwargs={'value': 'same'})
    due = NOW + timedelta(seconds=20)
    entry.request_run(due)
    stale = load(entry)
    load(entry).request_run(due)
    app.now = lambda: due

    next(stale)
    stored = load(entry)

    assert stored.last_run_at == due
    assert stored.total_run_count == 1
    assert not stored._strtobool(stored.run_immediately)
    assert set(app.redbeat_redis.hgetall(entry.key)) == {
        'definition',
        'meta',
        'jitter_fingerprint',
        'entry_id',
    }


@pytest.mark.parametrize('update_definition', [False, True])
def test_boolean_enabled_flag_remains_reservable_after_request_reload(app, update_definition):
    entry = create_entry(app)
    entry.enabled = True
    entry.request_run(NOW, run_immediately=True)
    if update_definition:
        current = load(entry)
        current.enabled = True
        current.kwargs = {'value': 'latest'}
        current.update_definition()
    current = load(entry)

    assert current.is_due()[0]
    outgoing = next(current)
    stored = load(entry)

    assert stored.total_run_count == 1
    assert stored._strtobool(stored.enabled)
    assert not stored._strtobool(stored.run_immediately)
    assert outgoing.kwargs == ({'value': 'latest'} if update_definition else {})


@pytest.mark.parametrize('reload_request', [False, True])
@pytest.mark.parametrize('enabled', [True, 'true', 'True'])
def test_request_reservation_compares_the_persisted_payload_representation(
    app, reload_request, enabled
):
    entry = create_entry(app, total_run_count=4)
    entry.args = ('argument',)
    entry.kwargs = {'values': (1, 2), 'observed_at': NOW}
    entry.enabled = enabled
    due = NOW + timedelta(seconds=20)
    entry.request_run(due)
    current = load(entry) if reload_request else entry
    app.now = lambda: due

    assert current.is_due()[0]
    next(current)
    stored = load(entry)

    assert stored.args == ['argument']
    assert stored.kwargs == {'values': [1, 2], 'observed_at': NOW}
    assert stored.last_run_at == due
    assert stored.total_run_count == 5
    assert not stored._strtobool(stored.run_immediately)


def test_disabled_entry_does_not_run_a_mature_request(app):
    entry = create_entry(app, enabled=False)
    entry.request_run(NOW, run_immediately=True)
    assert load(entry).is_due() == (False, 5.0)


def test_immediate_request_preserves_run_count(app):
    entry = create_entry(app, total_run_count=9)
    entry.request_run(NOW, run_immediately=True)
    assert load(entry).last_run_at == NOW - timedelta(seconds=60)
    assert load(entry).total_run_count == 9
    assert load(entry).is_due()[0]
    next(load(entry))
    assert load(entry).last_run_at == NOW
    assert load(entry).total_run_count == 10


def test_request_can_apply_new_interval_and_policy_together(app):
    entry = create_entry(app)
    entry.schedule = schedule(120, app=app)
    due = NOW + timedelta(seconds=20)
    with pytest.raises(ValueError, match='interval'):
        entry.request_run(due)
    entry.request_run(due, fingerprint='new-policy')
    stored = load(entry)
    assert stored.schedule.run_every == timedelta(seconds=120)
    assert stored.jitter_fingerprint == 'new-policy'
    app.now = lambda: due
    next(stored)
    assert (
        app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, entry.key)
        == (due + timedelta(seconds=120)).timestamp()
    )


def test_request_watch_conflict_rechecks_immediate_priority(app):
    entry = create_entry(app)
    immediate = load(entry)
    pipeline_class = type(app.redbeat_redis.pipeline())
    execute = pipeline_class.execute
    attempts = []

    def conflict_once(pipe, *args, **kwargs):
        attempts.append(True)
        if len(attempts) == 1:
            with patch.object(pipeline_class, 'execute', execute):
                immediate.request_run(NOW, run_immediately=True)
        return execute(pipe, *args, **kwargs)

    with patch.object(pipeline_class, 'execute', conflict_once):
        entry.request_run(NOW + timedelta(seconds=20))
    assert len(attempts) == 2
    assert load(entry)._strtobool(load(entry).run_immediately)
    assert load(entry).last_run_at == NOW - timedelta(seconds=60)
    assert load(entry).is_due()[0]


def test_request_stops_after_three_conflicts_and_does_not_retry_missing_entry(app):
    entry = create_entry(app)
    pipeline_class = type(app.redbeat_redis.pipeline())
    execute = pipeline_class.execute
    attempts = []

    def conflict(pipe, *args, **kwargs):
        attempts.append(True)
        app.redbeat_redis.hset(entry.key, 'concurrent_value', str(len(attempts)))
        return execute(pipe, *args, **kwargs)

    with patch.object(pipeline_class, 'execute', conflict):
        with pytest.raises(WatchError):
            entry.request_run(NOW, run_immediately=True)
    assert len(attempts) == 3
    assert not load(entry)._strtobool(load(entry).run_immediately)
    entry.delete()
    with patch.object(pipeline_class, 'execute') as commit:
        with pytest.raises(KeyError):
            entry.request_run(NOW, run_immediately=True)
    commit.assert_not_called()
