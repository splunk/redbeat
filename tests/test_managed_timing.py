from datetime import timedelta, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest
from celery.schedules import schedule

from redbeat.decoder import to_timestamp
from redbeat.schedulers import RedBeatScheduler, RedBeatSchedulerEntry
from tests.test_jitter_concurrent_updates import app as timing_app
from tests.test_jitter_concurrent_updates import load
from tests.test_jitter_integration import NOW
from tests.test_jitter_integration import app as redis_app  # noqa: F401

app = timing_app


@pytest.mark.parametrize(
    'interval,due',
    [
        (0.5, NOW + timedelta(microseconds=1)),
        (1.5, NOW + timedelta(seconds=2, microseconds=250000)),
        (60, NOW.replace(microsecond=999999).astimezone(timezone(timedelta(hours=5.5)))),
        (60, NOW.astimezone(timezone(timedelta(hours=-3.5)))),
        (3600, NOW.replace(month=3, day=8, hour=7).astimezone(ZoneInfo('America/New_York'))),
        (3600, NOW.replace(month=11, day=1, hour=6).astimezone(ZoneInfo('America/New_York'))),
    ],
    ids=['subsecond', 'fractional', 'east', 'west', 'spring-forward', 'fall-back'],
)
def test_managed_first_due_preserves_instant_across_reload_and_recurrence(app, interval, due):
    utc_due = due.astimezone(timezone.utc)
    interval = timedelta(seconds=interval)
    app.now = lambda: utc_due - timedelta(seconds=1)
    initial = RedBeatSchedulerEntry(
        'generic-timing', 'tasks.example', schedule(interval, app=app), app=app
    ).save_with_first_due(due, 'library-policy')
    initial.kwargs = {'revision': 'new'}
    initial.update_definition()
    stored = load(initial)
    assert stored.last_run_at == utc_due - interval
    assert stored.last_run_at.tzinfo == timezone.utc
    assert stored.schedule.run_every == interval
    assert stored.total_run_count == 0
    assert app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, initial.key) == to_timestamp(due)

    scheduler = RedBeatScheduler(app, lazy=True, max_interval=7200)
    try:
        with patch.object(app, 'send_task') as publish:
            for deadline in (utc_due, utc_due + interval):
                app.now = lambda: deadline - timedelta(microseconds=1)
                scheduler.tick()
                publish.assert_not_called()
                app.now = lambda: deadline
                scheduler.tick()
                publish.assert_called_once()
                assert publish.call_args.args[2] == {'revision': 'new'}
                publish.reset_mock()
        assert load(initial).total_run_count == 2
        assert load(initial).last_run_at == utc_due + interval
    finally:
        scheduler.close()


def test_fractional_requested_due_preserves_earliest_request_and_latest_payload(app):
    initial = RedBeatSchedulerEntry(
        'generic-request', 'tasks.example', schedule(1.5, app=app), app=app
    ).save_with_first_due(NOW, 'library-policy')
    next(initial)
    due = NOW + timedelta(seconds=20, microseconds=250000)
    for seconds, revision in ((0, 'first'), (10, 'latest')):
        current = load(initial)
        current.kwargs = {'revision': revision}
        current.request_run(
            (due + timedelta(seconds=seconds)).astimezone(timezone(timedelta(hours=5.5)))
        )
    stored = load(initial)
    assert stored.last_run_at == due - timedelta(seconds=1.5)
    assert stored.last_run_at.tzinfo == timezone.utc
    assert stored.kwargs == {'revision': 'latest'}
    scheduler = RedBeatScheduler(app, lazy=True, max_interval=60)
    try:
        with patch.object(app, 'send_task') as publish:
            app.now = lambda: due - timedelta(microseconds=1)
            scheduler.tick()
            publish.assert_not_called()
            app.now = lambda: due
            scheduler.tick()
            publish.assert_called_once()
        assert load(initial).total_run_count == 2
        assert load(initial).last_run_at == due
        assert not load(initial)._strtobool(load(initial).run_immediately)
    finally:
        scheduler.close()


@pytest.mark.parametrize('urgent_first', [False, True])
def test_fractional_urgent_request_wins_in_either_order(app, urgent_first):
    initial = RedBeatSchedulerEntry(
        'generic-urgent', 'tasks.example', schedule(1.5, app=app), app=app
    ).save_with_first_due(NOW, 'library-policy')
    due = NOW + timedelta(microseconds=125000)
    app.now = lambda: due
    for urgent in (urgent_first, not urgent_first):
        load(initial).request_run(
            due if urgent else due + timedelta(seconds=20.25), run_immediately=urgent
        )
    stored = load(initial)
    assert stored.last_run_at == due - timedelta(seconds=1.5)
    app.now = lambda: due - timedelta(microseconds=1)
    assert not stored.is_due()[0]
    app.now = lambda: due
    assert stored.is_due()[0]
    next(stored)
    assert load(initial).last_run_at == due
    assert load(initial).total_run_count == 1
    assert not load(initial)._strtobool(load(initial).run_immediately)


@pytest.mark.parametrize('due', [NOW.replace(tzinfo=None), None])
def test_invalid_requested_due_keeps_saved_state(app, due):
    initial = RedBeatSchedulerEntry(
        'invalid-request', 'tasks.example', schedule(1.5, app=app), app=app
    ).save_with_first_due(NOW, 'library-policy')
    before = app.redbeat_redis.hgetall(initial.key)
    score = app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, initial.key)
    with pytest.raises(ValueError, match='timezone-aware'):
        initial.request_run(due)
    assert app.redbeat_redis.hgetall(initial.key) == before
    assert app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, initial.key) == score


def test_naive_following_deadline_keeps_current_run_and_ordinary_recurrence(app):
    initial = RedBeatSchedulerEntry(
        'invalid-policy', 'tasks.example', schedule(1.5, app=app), app=app
    ).save_with_first_due(NOW, 'library-policy')
    app.conf.redbeat_entry_policy = lambda following: (
        NOW.replace(tzinfo=None) + timedelta(seconds=3),
        'rejected-policy',
    )
    outgoing = next(initial)
    assert outgoing.last_run_at == NOW
    assert load(initial).last_run_at == NOW
    assert load(initial).total_run_count == 1
    assert load(initial).jitter_fingerprint == 'library-policy'
    app.now = lambda: NOW + timedelta(seconds=1.5)
    assert load(initial).is_due()[0]


@pytest.mark.parametrize('managed', [False, True])
def test_following_policy_normalizes_fractional_due_and_keeps_current_run(app, managed):
    initial = RedBeatSchedulerEntry(
        'generic-policy', 'tasks.example', schedule(1.5, app=app), app=app
    )
    if managed:
        initial.save_with_first_due(NOW, 'library-policy')
    else:
        initial.save()
    due = NOW + timedelta(seconds=3, microseconds=250000)

    def policy(entry):
        if entry.jitter_fingerprint != 'library-policy-two':
            return due.astimezone(timezone(timedelta(hours=-3.5))), 'library-policy-two'
        return None

    app.conf.redbeat_entry_policy = policy
    outgoing = next(load(initial))
    assert outgoing.last_run_at == NOW
    assert load(initial).total_run_count == 1
    assert load(initial).last_run_at == due - timedelta(seconds=1.5)
    assert load(initial).last_run_at.tzinfo == timezone.utc
    assert load(initial).jitter_fingerprint == 'library-policy-two'
    scheduler = RedBeatScheduler(app, lazy=True, max_interval=60)
    try:
        with patch.object(app, 'send_task') as publish:
            for deadline in (due, due + timedelta(seconds=1.5)):
                app.now = lambda: deadline - timedelta(microseconds=1)
                scheduler.tick()
                publish.assert_not_called()
                app.now = lambda: deadline
                scheduler.tick()
                publish.assert_called_once()
                publish.reset_mock()
        assert load(initial).total_run_count == 3
    finally:
        scheduler.close()
