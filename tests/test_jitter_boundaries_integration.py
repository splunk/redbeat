from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from threading import Barrier
from unittest.mock import patch

import pytest
from celery.schedules import schedule
from redis.exceptions import WatchError

from redbeat.schedulers import RedBeatScheduler, RedBeatSchedulerEntry
from tests.test_jitter_integration import FINGERPRINT, NOW
from tests.test_jitter_integration import app as redis_app
from tests.test_jitter_integration import read_message

app = redis_app


def create_entry(app, interval=60, delay=20):
    app.now = lambda: NOW
    entry = RedBeatSchedulerEntry(
        'first-due-test', 'validation.noop', schedule(interval, app=app), app=app
    )
    entry.save_with_first_due(NOW + timedelta(seconds=delay), FINGERPRINT)
    return RedBeatSchedulerEntry.from_key(entry.key, app=app)


@pytest.mark.parametrize('interval', [1, 10, 60, 300, 301, 600, 1800, 86400])
@pytest.mark.parametrize('delay', [1, 20, 600])
def test_supplied_first_due_and_three_reservations(app, interval, delay):
    entry = create_entry(app, interval, delay)
    first = NOW + timedelta(seconds=delay)
    assert entry.last_run_at == first - timedelta(seconds=interval)
    assert app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, entry.key) == first.timestamp()
    scheduler = RedBeatScheduler(app, lazy=True, max_interval=86400)
    with app.connection_for_read() as connection:
        queue = connection.SimpleQueue(app.conf.task_default_queue)
        try:
            for cycle in range(3):
                due = first + timedelta(seconds=cycle * interval)
                app.now = lambda: due - timedelta(microseconds=1)
                scheduler.tick()
                assert read_message(queue) is None
                app.now = lambda: due
                scheduler.tick()
                message = read_message(queue)
                assert message is not None
                message.ack()
                assert read_message(queue) is None
                stored = RedBeatSchedulerEntry.from_key(entry.key, app=app)
                assert stored.total_run_count == cycle + 1
                assert stored.jitter_fingerprint == FINGERPRINT
                assert (
                    app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, entry.key)
                    == (due + timedelta(seconds=interval)).timestamp()
                )
        finally:
            queue.close()
            scheduler.close()


def test_long_outage_publishes_once_and_reanchors(app):
    entry = create_entry(app, interval=10)
    late = NOW + timedelta(hours=2)
    app.now = lambda: late
    scheduler = RedBeatScheduler(app, lazy=True)
    with patch.object(app, 'send_task') as publish:
        for _ in range(5):
            scheduler.tick()
    assert publish.call_count == 1
    assert RedBeatSchedulerEntry.from_key(entry.key, app=app).last_run_at == late
    assert (
        app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, entry.key)
        == (late + timedelta(seconds=10)).timestamp()
    )


def test_fractional_reservation_does_not_publish_early(app):
    entry = create_entry(app, interval=10, delay=1)
    late = NOW + timedelta(seconds=11, microseconds=900000)
    app.now = lambda: late
    scheduler = RedBeatScheduler(app, lazy=True)
    with app.connection_for_read() as connection:
        queue = connection.SimpleQueue(app.conf.task_default_queue)
        try:
            scheduler.tick()
            message = read_message(queue)
            assert message is not None
            message.ack()
            next_due = late + timedelta(seconds=10)
            assert app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, entry.key) == int(
                next_due.timestamp()
            )
            app.now = lambda: next_due - timedelta(microseconds=1)
            scheduler.tick()
            assert read_message(queue) is None
            app.now = lambda: next_due
            scheduler.tick()
            message = read_message(queue)
            assert message is not None
            message.ack()
        finally:
            queue.close()
            scheduler.close()


@pytest.mark.parametrize('orphan', [False, True])
def test_exact_creation_refuses_existing_hash_without_repair(app, orphan):
    entry = create_entry(app)
    if orphan:
        app.redbeat_redis.hdel(entry.key, 'definition')
    before = app.redbeat_redis.hgetall(entry.key)
    score = app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, entry.key)
    with pytest.raises(ValueError, match='existing schedule hash'):
        entry.save_with_first_due(NOW + timedelta(seconds=900), entry.jitter_fingerprint)
    assert app.redbeat_redis.hgetall(entry.key) == before
    assert app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, entry.key) == score


def test_publication_failure_keeps_existing_reservation_boundary(app):
    entry = create_entry(app)
    due = NOW + timedelta(seconds=20)
    app.now = lambda: due
    scheduler = RedBeatScheduler(app, lazy=True)
    with patch.object(app, 'send_task', side_effect=OSError('synthetic broker failure')):
        scheduler.tick()
    stored = RedBeatSchedulerEntry.from_key(entry.key, app=app)
    assert stored.total_run_count == 1
    assert stored.last_run_at == due
    assert (
        app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, entry.key)
        == (due + timedelta(seconds=60)).timestamp()
    )


def test_competing_creators_store_one_consistent_winner(app):
    """
    Race isolated creators and check that only one complete state is accepted.

    :param app: App connected to disposable Redis

    :return: None
    """
    app.now = lambda: NOW
    start = Barrier(8)

    def create(number):
        entry = RedBeatSchedulerEntry(
            name='competing-creators',
            task='validation.noop',
            schedule=schedule(60, app=app),
            kwargs={'creator': number},
            app=app,
        )
        due = NOW + timedelta(seconds=number + 1)
        start.wait(timeout=10)
        try:
            entry.save_with_first_due(due, f'creator-{number}')
        except (ValueError, WatchError):
            return None
        return number

    with ThreadPoolExecutor(max_workers=8) as executor:
        winners = [n for n in executor.map(create, range(8)) if n is not None]
    assert len(winners) == 1
    winner = winners[0]
    key = 'redbeat:competing-creators'
    saved = RedBeatSchedulerEntry.from_key(key, app=app)
    assert saved.kwargs == {'creator': winner}
    assert saved.jitter_fingerprint == f'creator-{winner}'
    due = NOW + timedelta(seconds=winner + 1)
    assert saved.last_run_at == due - timedelta(seconds=60)
    assert app.redbeat_redis.zscore(app.redbeat_conf.schedule_key, key) == due.timestamp()
    assert not app.redbeat_redis.sismember('run_immediately', key)


def test_first_due_removes_stale_immediate_membership(app):
    key = 'redbeat:first-due-test'
    app.redbeat_redis.sadd('run_immediately', key, 'unrelated-entry')
    entry = create_entry(app)
    assert entry.key == key
    assert not app.redbeat_redis.sismember('run_immediately', key)
    assert app.redbeat_redis.sismember('run_immediately', 'unrelated-entry')
