from datetime import timedelta
from unittest.mock import Mock, patch

import pytest
from celery.schedules import schedule
from redis.exceptions import WatchError

from redbeat.schedulers import EntryPolicyChanged, RedBeatScheduler, RedBeatSchedulerEntry
from tests.test_jitter_concurrent_updates import app as policy_app
from tests.test_jitter_concurrent_updates import create_entry, load
from tests.test_jitter_integration import NOW
from tests.test_jitter_integration import app as redis_app  # noqa: F401

app = policy_app


def score(entry):
    return entry.app.redbeat_redis.zscore(entry.app.redbeat_conf.schedule_key, entry.key)


def legacy_entry(app, **kwargs):
    return RedBeatSchedulerEntry(
        'ordinary-entry', 'tasks.test', schedule(60, app=app), app=app, **kwargs
    ).save()


def phase_following(entry):
    return entry.last_run_at + timedelta(seconds=80), 'policy-one'


def test_policy_preserves_current_due_and_runs_only_during_reservation(app):
    entry = create_entry(app, NOW + timedelta(seconds=60))
    before = app.redbeat_redis.hgetall(entry.key)
    callback = Mock(side_effect=phase_following)
    app.conf.redbeat_entry_policy = callback
    scheduler = RedBeatScheduler(app, lazy=True, max_interval=3600)
    try:
        with patch.object(app, 'send_task') as publish:
            for instant in (NOW, NOW + timedelta(seconds=59, microseconds=999999)):
                app.now = lambda: instant
                scheduler.tick()
                callback.assert_not_called()
                publish.assert_not_called()
                assert app.redbeat_redis.hgetall(entry.key) == before
                assert score(entry) == (NOW + timedelta(seconds=60)).timestamp()
            app.now = lambda: NOW + timedelta(seconds=60)
            scheduler.tick()
            publish.assert_called_once()
        callback.assert_called_once()
        assert score(entry) == (NOW + timedelta(seconds=140)).timestamp()
        assert load(entry).total_run_count == 1
    finally:
        scheduler.close()


@pytest.mark.parametrize('immediate', [False, True])
def test_first_adoption_serves_current_ordinary_run_then_phases_following(app, immediate):
    entry = legacy_entry(app, run_immediately=immediate, kwargs={'value': 'original'})
    app.conf.redbeat_entry_policy = phase_following
    scheduler = RedBeatScheduler(app, lazy=True, max_interval=5)
    try:
        if not immediate:
            app.now = lambda: NOW + timedelta(seconds=60)
        with patch.object(app, 'send_task') as publish:
            scheduler.tick()
        publish.assert_called_once()
        stored = load(entry)
        assert stored.jitter_fingerprint == 'policy-one'
        assert stored.kwargs == {'value': 'original'}
        assert stored.total_run_count == 1
        assert stored.last_run_at == app.now() + timedelta(seconds=20)
        assert score(entry) == (app.now() + timedelta(seconds=80)).timestamp()
        assert not stored._strtobool(stored.run_immediately)
        assert not app.redbeat_redis.sismember('run_immediately', entry.key)
    finally:
        scheduler.close()


def test_policy_sees_fresh_recurring_payload_and_actual_reservation_metadata(app):
    entry = create_entry(app, kwargs={'value': 'old'}, first_run_kwargs={'bootstrap': 'first'})
    current = load(entry)
    current.kwargs = {'value': 'new'}
    current.args = ['latest']
    current.options = {'queue': 'latest-queue', 'expires': 120}
    current.update_definition()
    calls = []

    def policy(following):
        calls.append(
            (
                dict(following.kwargs),
                following.args,
                following.last_run_at,
                following.total_run_count,
                dict(following.first_run_kwargs),
            )
        )
        return phase_following(following)

    app.conf.redbeat_entry_policy = policy
    outgoing = next(entry)
    assert calls == [({'value': 'new'}, ['latest'], NOW, 1, {})]
    assert outgoing.kwargs == {'value': 'old', 'bootstrap': 'first'}
    assert outgoing.last_run_at == NOW
    assert outgoing.total_run_count == 1
    assert load(entry).kwargs == {'value': 'new'}
    assert load(entry).first_run_kwargs == {}
    assert load(entry).last_run_at == NOW + timedelta(seconds=20)


@pytest.mark.parametrize('managed', [False, True])
def test_no_proposal_keeps_normal_reservation_and_no_new_fields(app, managed):
    entry = create_entry(app) if managed else legacy_entry(app)
    callback = Mock(return_value=None)
    app.conf.redbeat_entry_policy = callback
    before = app.redbeat_redis.hget(entry.key, 'definition')
    if managed:
        outgoing = next(load(entry))
    else:
        pipeline_class = type(app.redbeat_redis.pipeline())
        with patch.object(pipeline_class, 'watch') as watched:
            outgoing = next(load(entry))
        watched.assert_not_called()
    callback.assert_called_once()
    assert outgoing.last_run_at == NOW
    assert load(entry).last_run_at == NOW
    assert load(entry).total_run_count == 1
    assert score(entry) == (NOW + timedelta(seconds=60)).timestamp()
    assert app.redbeat_redis.hget(entry.key, 'definition') == before
    assert set(app.redbeat_redis.hgetall(entry.key)) == (
        {'definition', 'meta', 'jitter_fingerprint', 'entry_id'}
        if managed
        else {'definition', 'meta'}
    )


def test_metadata_only_update_does_not_call_policy(app):
    entry = create_entry(app)
    callback = Mock(side_effect=phase_following)
    app.conf.redbeat_entry_policy = callback
    outgoing = entry.next(only_update_last_run_at=True)
    callback.assert_not_called()
    assert outgoing.total_run_count == load(entry).total_run_count == 0
    assert load(entry).last_run_at == NOW


@pytest.mark.parametrize('fingerprint', ['policy-two', None])
def test_fingerprint_only_proposal_keeps_normal_following_due(app, fingerprint):
    entry = create_entry(app)
    app.conf.redbeat_entry_policy = lambda following: (None, fingerprint)
    next(entry)
    assert load(entry).jitter_fingerprint == fingerprint
    assert load(entry).last_run_at == NOW
    assert load(entry).total_run_count == 1
    assert score(entry) == (NOW + timedelta(seconds=60)).timestamp()


def test_disable_removes_policy_during_reservation_and_allows_ordinary_updates(app):
    entry = create_entry(app)
    app.conf.redbeat_entry_policy = lambda following: (NOW + timedelta(seconds=60), None)
    next(entry)
    updated = load(entry)
    assert updated.jitter_fingerprint is None
    app.conf.redbeat_entry_policy = None
    updated.schedule = schedule(120, app=app)
    updated.run_immediately = True
    updated = updated.save_guarded()
    assert updated.last_run_at == NOW
    assert updated.total_run_count == 1
    assert updated.is_due()[0]
    next(updated)
    assert load(updated).total_run_count == 2
    assert score(updated) == (NOW + timedelta(seconds=120)).timestamp()
    with pytest.raises(EntryPolicyChanged):
        entry.update_definition()


def test_loaded_legacy_save_cannot_overwrite_first_policy_adoption(app):
    stale = legacy_entry(app)
    app.conf.redbeat_entry_policy = phase_following
    next(load(stale))
    before = app.redbeat_redis.hgetall(stale.key)
    stale.kwargs = {'value': 'new'}
    with pytest.raises(EntryPolicyChanged):
        stale.save_guarded()
    assert app.redbeat_redis.hgetall(stale.key) == before
    assert score(stale) == (NOW + timedelta(seconds=80)).timestamp()


@pytest.mark.parametrize('run_immediately', [False, True])
def test_guarded_save_rejects_managed_pending_request_without_writes(app, run_immediately):
    entry = create_entry(app, kwargs={'value': 'accepted'})
    entry.request_run(NOW + timedelta(seconds=20), run_immediately=run_immediately)
    entry = load(entry)
    before = (app.redbeat_redis.hgetall(entry.key), score(entry))
    entry.kwargs = {'value': 'rejected'}
    with pytest.raises(ValueError, match='Managed entries require'):
        entry.save_guarded()
    assert (app.redbeat_redis.hgetall(entry.key), score(entry)) == before


@pytest.mark.parametrize('during_watch', [False, True])
def test_first_adoption_keeps_new_ordinary_immediate_request(app, during_watch):
    entry = legacy_entry(app, kwargs={'value': 'old'})
    current = load(entry)
    calls = []

    def policy(following):
        calls.append(True)
        if len(calls) == (2 if during_watch else 1):
            current.kwargs = {'value': 'new'}
            current.run_immediately = True
            current.save_guarded()
        return phase_following(following)

    app.conf.redbeat_entry_policy = policy
    outgoing = next(entry)
    stored = load(entry)
    assert outgoing.kwargs == {'value': 'old'}
    assert stored.kwargs == {'value': 'new'}
    assert stored.jitter_fingerprint is None
    assert stored._strtobool(stored.run_immediately)
    assert app.redbeat_redis.sismember('run_immediately', entry.key)
    scheduler = RedBeatScheduler(app, lazy=True, max_interval=1)
    try:
        with patch.object(app, 'send_task') as publish:
            scheduler.tick()
        publish.assert_called_once()
        assert load(entry).jitter_fingerprint == 'policy-one'
        assert not load(entry)._strtobool(load(entry).run_immediately)
    finally:
        scheduler.close()


def test_new_first_run_context_is_preserved_before_policy_adoption(app):
    stale = create_entry(app, first_run_kwargs={'bootstrap': 'old'})
    current = load(stale)
    current.first_run_kwargs = {'bootstrap': 'new'}
    current.save()
    callback = Mock(side_effect=phase_following)
    app.conf.redbeat_entry_policy = callback
    outgoing = next(stale)
    callback.assert_not_called()
    assert outgoing.kwargs == {'bootstrap': 'old'}
    assert load(stale).first_run_kwargs == {'bootstrap': 'new'}
    assert load(stale).jitter_fingerprint == stale.jitter_fingerprint


def test_policy_watch_retry_reads_latest_payload_without_changing_outgoing(app):
    entry = create_entry(app, kwargs={'value': 'old'})
    current = load(entry)
    calls = []

    def policy(following):
        calls.append(following.kwargs['value'])
        if len(calls) == 1:
            current.kwargs = {'value': 'new'}
            current.update_definition()
        return phase_following(following)

    app.conf.redbeat_entry_policy = policy
    outgoing = next(entry)
    assert calls == ['old', 'new']
    assert outgoing.kwargs == {'value': 'old'}
    assert load(entry).kwargs == {'value': 'new'}
    assert score(entry) == (NOW + timedelta(seconds=80)).timestamp()


def test_new_managed_request_during_policy_is_not_erased_or_published_stale(app):
    entry = create_entry(app)
    current = load(entry)

    def policy(following):
        current.kwargs = {'value': 'accepted'}
        current.request_run(NOW + timedelta(seconds=20))
        return phase_following(following)

    app.conf.redbeat_entry_policy = policy
    scheduler = RedBeatScheduler(app, lazy=True, max_interval=1)
    try:
        with patch.object(app, 'send_task') as publish:
            scheduler.tick()
        publish.assert_not_called()
        assert load(entry).total_run_count == 0
        assert load(entry).kwargs == {'value': 'accepted'}
        assert load(entry)._strtobool(load(entry).run_immediately)
        assert score(entry) == (NOW + timedelta(seconds=20)).timestamp()
    finally:
        scheduler.close()


@pytest.mark.parametrize('managed', [False, True])
@pytest.mark.parametrize('failure', ['raises', 'shape', 'fingerprint', 'due', 'past', 'equal'])
def test_policy_failure_keeps_current_publication(app, managed, failure):
    entry = create_entry(app) if managed else legacy_entry(app, run_immediately=True)
    original_fingerprint = entry.jitter_fingerprint

    def policy(following):
        if failure == 'raises':
            raise ValueError('application policy failure')
        if failure == 'shape':
            return (NOW,)
        if failure == 'fingerprint':
            return NOW + timedelta(seconds=80), ''
        if failure == 'past':
            return NOW - timedelta(seconds=1), 'policy-two'
        if failure == 'equal':
            return NOW, 'policy-two'
        return NOW.replace(tzinfo=None), 'policy-two'

    app.conf.redbeat_entry_policy = policy
    scheduler = RedBeatScheduler(app, lazy=True, max_interval=1)
    try:
        with patch.object(app, 'send_task') as publish:
            scheduler.tick()
        publish.assert_called_once()
        assert load(entry).total_run_count == 1
        assert load(entry).jitter_fingerprint == original_fingerprint
        assert score(entry) == (NOW + timedelta(seconds=60)).timestamp()
    finally:
        scheduler.close()


def test_publish_failure_keeps_existing_reservation_boundary(app):
    entry = create_entry(app)
    app.conf.redbeat_entry_policy = phase_following
    scheduler = RedBeatScheduler(app, lazy=True, max_interval=1)
    try:
        with patch.object(
            app, 'send_task', side_effect=RuntimeError('broker unavailable')
        ) as publish:
            scheduler.tick()
        publish.assert_called_once()
        assert load(entry).total_run_count == 1
        assert load(entry).jitter_fingerprint == 'policy-one'
        assert score(entry) == (NOW + timedelta(seconds=80)).timestamp()
    finally:
        scheduler.close()


@pytest.mark.parametrize('conflicts', [1, 3])
def test_following_policy_commit_has_at_most_three_watch_attempts(app, conflicts):
    entry = create_entry(app)
    before = app.redbeat_redis.hgetall(entry.key)
    before_score = score(entry)
    app.conf.redbeat_entry_policy = phase_following
    pipeline_class = type(app.redbeat_redis.pipeline())
    execute = pipeline_class.execute
    attempts = []

    def conflict(pipe, *args, **kwargs):
        if any(command[0][0] == 'ZADD' for command in pipe.command_stack):
            attempts.append(True)
            if len(attempts) <= conflicts:
                raise WatchError('concurrent reservation')
        return execute(pipe, *args, **kwargs)

    with patch.object(pipeline_class, 'execute', conflict):
        if conflicts == 3:
            with pytest.raises(WatchError):
                next(entry)
            assert app.redbeat_redis.hgetall(entry.key) == before
            assert score(entry) == before_score
        else:
            next(entry)
            assert score(entry) == (NOW + timedelta(seconds=80)).timestamp()
    assert len(attempts) == min(conflicts + 1, 3)
