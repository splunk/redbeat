# Licensed under the Apache License, Version 2.0 (the 'License'); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
# Copyright 2015 Marc Sibson


import json
import ssl
import warnings
from datetime import MINYEAR, datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

import redis.exceptions
from celery.app import app_or_default
from celery.beat import DEFAULT_MAX_INTERVAL, ScheduleEntry, Scheduler
from celery.signals import beat_init
from celery.utils.log import get_logger
from celery.utils.time import humanize_seconds
from kombu.utils.objects import cached_property
from kombu.utils.url import maybe_sanitize_url
from redis import Redis
from redis.sentinel import MasterNotFoundError, Sentinel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
)

from .decoder import RedBeatJSONDecoder, RedBeatJSONEncoder, to_timestamp

logger = get_logger('celery.beat')

# Copied from:
# https://github.com/andymccurdy/redis-py/blob/master/redis/lock.py#L33
# Changes:
#     The second line from the bottom: The original Lua script intends
#     to extend time to (lock remaining time + additional time); while
#     the script here extend time to a expected expiration time.
# KEYS[1] - lock name
# ARGS[1] - token
# ARGS[2] - additional milliseconds
# return 1 if the locks time was extended, otherwise 0
LUA_EXTEND_TO_SCRIPT = """
    local token = redis.call('get', KEYS[1])
    if not token or token ~= ARGV[1] then
        return 0
    end
    local expiration = redis.call('pttl', KEYS[1])
    if not expiration then
        expiration = 0
    end
    if expiration < 0 then
        return 0
    end
    redis.call('pexpire', KEYS[1], ARGV[2])
    return 1
"""

REDBEAT_REDIS_KEY = "redbeat_redis"
REDBEAT_SENTINEL_KEY = "redbeat_sentinel"

# Use three attempts in total for Redis write conflicts.
# Policy changes are raised separately so the caller can reload the entry.
_retry_watch_conflicts = retry(
    retry=retry_if_exception_type(redis.exceptions.WatchError),
    stop=stop_after_attempt(3),
    reraise=True,
)


class EntryPolicyChanged(ValueError):
    """The saved schedule has a different applied policy."""


class RetryingConnection:
    """A proxy for the Redis connection that delegates all the calls to
    underlying Redis connection while retrying on connection or time-out error.
    """

    RETRY_MAX_WAIT = 30

    def __init__(self, retry_period, wrapped_connection):
        self.wrapped_connection = wrapped_connection
        self.retry_kwargs = dict(
            retry=(
                retry_if_exception_type(redis.exceptions.ConnectionError)
                | retry_if_exception_type(redis.exceptions.TimeoutError)
                | retry_if_exception_type(MasterNotFoundError)
            ),
            reraise=True,
            wait=wait_exponential(multiplier=1, max=self.RETRY_MAX_WAIT),
            before_sleep=self._log_retry_attempt,
        )
        if retry_period >= 0:
            self.retry_kwargs.update(dict(stop=stop_after_delay(retry_period)))

    def __getattr__(self, item):
        method = getattr(self.wrapped_connection, item)

        # we don't want to deal attributes or properties
        if not callable(method):
            return method

        @retry(**self.retry_kwargs)
        def retrier(*args, **kwargs):
            return method(*args, **kwargs)

        return retrier

    @staticmethod
    def _log_retry_attempt(retry_state):
        """Log when next reconnection attempt is about to be made."""
        logger.warning(
            'beat: Retrying Redis connection in %s seconds...', retry_state.next_action.sleep
        )


def ensure_conf(app):
    """
    Ensure for the given app the the redbeat_conf
    attribute is set to an instance of the RedBeatConfig
    class.
    """
    name = 'redbeat_conf'
    app = app_or_default(app)
    try:
        config = getattr(app, name)
    except AttributeError:
        config = RedBeatConfig(app)
        setattr(app, name, config)

    return config


# Options consumed by RedBeat itself, never forwarded to the redis client.
REDBEAT_INTERNAL_OPTIONS = (
    'retry_period',
    'cluster',
    'sentinels',
    'service_name',
    'sentinel_kwargs',
    'startup_nodes',
)


def get_redis(app=None):
    app = app_or_default(app)
    conf = ensure_conf(app)
    redis_options = dict(conf.redbeat_redis_options)
    retry_period = redis_options.pop('retry_period', None)

    passthrough_options = {
        key: value for key, value in redis_options.items() if key not in REDBEAT_INTERNAL_OPTIONS
    }
    if conf.is_key_in_conf('redbeat_redis_options'):
        # options were addressed to redbeat: pass everything except redbeat's
        # own keys through to the client, which validates its own arguments
        connection_options = passthrough_options
        skipped_options = {}
    else:
        # options inherited from broker_transport_options belong to the broker
        # and are not forwarded to the client, as before 2.4.0; the sentinel
        # and cluster branches below still consume them
        connection_options = {}
        skipped_options = passthrough_options

    if not hasattr(app, REDBEAT_REDIS_KEY) or getattr(app, REDBEAT_REDIS_KEY) is None:
        if redis_options.get('cluster', False):
            from redis.cluster import RedisCluster

            connection = RedisCluster.from_url(conf.redis_url, **passthrough_options)
        elif conf.redis_url.startswith('redis-sentinel') and 'sentinels' in redis_options:
            connection_kwargs = {}
            if isinstance(conf.redis_use_ssl, dict):
                connection_kwargs['ssl'] = True
                connection_kwargs.update(conf.redis_use_ssl)

            # Pass username via connection_kwargs rather than as a direct named argument
            # because redis<3.4.0 does not support username in the Sentinel constructor,
            # and redbeat supports redis>=3.2.
            username = redis_options.get('username')
            if username is not None:
                connection_kwargs['username'] = username

            sentinel = Sentinel(
                redis_options['sentinels'],
                socket_timeout=redis_options.get('socket_timeout'),
                password=redis_options.get('password'),
                db=redis_options.get('db', 0),
                decode_responses=True,
                sentinel_kwargs=redis_options.get('sentinel_kwargs'),
                credential_provider=redis_options.get('credential_provider'),
                **connection_kwargs,
            )
            setattr(app, REDBEAT_SENTINEL_KEY, sentinel)
            connection = None
        elif conf.redis_url.startswith('rediss'):
            _log_skipped_broker_options(skipped_options)
            ssl_options = {'ssl_cert_reqs': ssl.CERT_REQUIRED}
            if isinstance(conf.redis_use_ssl, dict):
                ssl_options.update(conf.redis_use_ssl)
            extras = {"decode_responses": True, **ssl_options, **connection_options}
            connection = Redis.from_url(conf.redis_url, **extras)
        elif conf.redis_url.startswith('redis-cluster'):
            from redis.cluster import RedisCluster

            startup_nodes = redis_options.get('startup_nodes') or [
                {"host": "localhost", "port": 30001}
            ]
            startup_nodes = [{**node, "port": int(node["port"])} for node in startup_nodes]

            passthrough_options.update({"decode_responses": True})
            connection = RedisCluster(startup_nodes=startup_nodes, **passthrough_options)
        else:
            _log_skipped_broker_options(skipped_options)
            connection_options.update({"decode_responses": True})
            connection = Redis.from_url(conf.redis_url, **connection_options)

        if connection:
            _set_redbeat_connect(app, REDBEAT_REDIS_KEY, connection, retry_period)

    if hasattr(app, REDBEAT_SENTINEL_KEY) and isinstance(
        getattr(app, REDBEAT_SENTINEL_KEY), Sentinel
    ):
        sentinel = getattr(app, REDBEAT_SENTINEL_KEY)
        connection = sentinel.master_for(
            redis_options.get('service_name', 'master'), db=redis_options.get('db', 0)
        )
        _set_redbeat_connect(app, REDBEAT_REDIS_KEY, connection, retry_period)

    return getattr(app, REDBEAT_REDIS_KEY)


def _log_skipped_broker_options(skipped_options):
    if skipped_options:
        logger.debug(
            'beat: not forwarding broker_transport_options to redis: %s; '
            'set redbeat_redis_options to pass options to the redis client',
            ', '.join(sorted(skipped_options)),
        )


def _set_redbeat_connect(app, connect_name, connection, retry_period):
    if retry_period is None:
        setattr(app, connect_name, connection)
    else:
        setattr(app, connect_name, RetryingConnection(retry_period, connection))


ADD_ENTRY_ERROR = """\

Couldn't add entry %r to redis schedule: %r. Contents: %r
"""


class RedBeatConfig:
    def __init__(self, app=None):
        self.app = app_or_default(app)
        self.key_prefix = self.either_or('redbeat_key_prefix', 'redbeat:')
        self.schedule_key = self.key_prefix + ':schedule'
        self.statics_key = self.key_prefix + ':statics'
        self.redis_url = self.either_or('redbeat_redis_url', app.conf['BROKER_URL'])
        if not self.is_key_in_conf('redbeat_redis_url'):
            # also log it: DeprecationWarning is silenced by default and beat
            # usually runs as a daemon, so the warning alone would go unseen
            message = (
                'RedBeat will stop falling back to broker_url in version 2.5.0, '
                'set redbeat_redis_url explicitly'
            )
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            logger.warning(message)
        self.redis_use_ssl = self.either_or('redbeat_redis_use_ssl', app.conf['BROKER_USE_SSL'])
        self.redbeat_redis_options = self.either_or(
            'redbeat_redis_options', app.conf['BROKER_TRANSPORT_OPTIONS']
        )
        if not self.is_key_in_conf('redbeat_redis_options') and self.redbeat_redis_options:
            message = (
                'RedBeat will stop falling back to broker_transport_options in version '
                '2.5.0, set redbeat_redis_options explicitly'
            )
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            logger.warning(message)
        self.lock_key = None
        self.lock_timeout = self.either_or('redbeat_lock_timeout', None)

    @property
    def schedule(self):
        return self.app.conf.beat_schedule

    @schedule.setter
    def schedule(self, value):
        self.app.conf.beat_schedule = value

    def either_or(self, name, default=None):
        if name == name.upper():
            warnings.warn(
                'Celery v4 installed, but detected Celery v3 '
                'configuration %s (use %s instead).' % (name, name.lower()),
                UserWarning,
            )
        if self.is_key_in_conf(name):
            return self.app.conf.first(name, name.upper())
        else:
            return self.app.conf.first(name, name.upper()) or default

    def is_key_in_conf(self, name):
        if name.upper() in map(lambda k: k.upper(), self.app.conf.keys()):
            return True


class RedBeatSchedulerEntry(ScheduleEntry):
    _meta = None

    def __init__(
        self,
        name=None,
        task=None,
        schedule=None,
        args=None,
        kwargs=None,
        enabled=True,
        options=None,
        target=None,
        run_immediately=False,
        jitter_fingerprint=None,
        first_run_kwargs=None,
        entry_id=None,
        **clsargs,
    ):
        super().__init__(
            name=str(name),
            task=task,
            schedule=schedule,
            args=args,
            kwargs=kwargs,
            options=options,
            **clsargs,
        )
        self.run_immediately = str(run_immediately)
        self.enabled = str(enabled)
        self.target = target
        self.jitter_fingerprint = jitter_fingerprint
        self.first_run_kwargs = dict(first_run_kwargs or {})
        self.entry_id = entry_id
        ensure_conf(self.app)

    @staticmethod
    def load_definition(key, app=None, definition=None):
        if definition is None:
            definition = get_redis(app).hget(key, 'definition')

        if not definition:
            raise KeyError(key)

        definition = RedBeatSchedulerEntry.decode_definition(definition)

        return definition

    @staticmethod
    def decode_definition(definition):
        return json.loads(definition, cls=RedBeatJSONDecoder)

    @staticmethod
    def decode_json(value):
        return json.loads(value) if value else None

    @staticmethod
    def load_meta(key, app=None):
        return RedBeatSchedulerEntry.decode_meta(get_redis(app).hget(key, 'meta'))

    @staticmethod
    def decode_meta(meta, app=None):
        if not meta:
            return {'last_run_at': None}

        return json.loads(meta, cls=RedBeatJSONDecoder)

    @staticmethod
    def generate_key(app: Any, name: str) -> str:
        return app.redbeat_conf.key_prefix + name

    @classmethod
    def from_key(cls, key, app=None):
        ensure_conf(app)
        with get_redis(app).pipeline() as pipe:
            pipe.hget(key, 'definition')
            pipe.hget(key, 'meta')
            pipe.hget(key, 'jitter_fingerprint')
            pipe.hget(key, 'entry_id')
            definition, meta, fingerprint, entry_id = pipe.execute()
        entry_id = cls.decode_json(entry_id)
        fingerprint = cls.decode_json(fingerprint)

        if not definition:
            raise KeyError(key)

        definition = cls.decode_definition(definition)
        meta = cls.decode_meta(meta)
        definition.update(meta)

        # Keep the policy marker and creation identity outside the task definition.
        entry = cls(
            app=app,
            jitter_fingerprint=fingerprint,
            entry_id=entry_id,
            **definition,
        )
        # celery.ScheduleEntry sets last_run_at = utcnow(), which is confusing and wrong
        entry.last_run_at = meta['last_run_at']

        return entry

    @classmethod
    def get_schedules_by_target(cls, target, app=None):
        ensure_conf(app)
        connection = get_redis(app)
        schedules_keys = connection.scan_iter(f"*{target};*")
        schedule_objects = []
        for key in schedules_keys:
            connection = get_redis(app)
            schedule_obj = connection.hgetall(key)
            definition = cls.decode_definition(schedule_obj["definition"])
            meta = cls.decode_definition(schedule_obj["meta"])
            entry = cls(app=app, **definition)
            entry.last_run_at = meta['last_run_at']
            schedule_objects.append(entry)

        return schedule_objects

    @classmethod
    def delete_schedules_by_target(cls, target, app=None):
        ensure_conf(app)
        connection = get_redis(app)
        schedules_keys = connection.scan_iter(f"*{target};*")
        for schedule_key in schedules_keys:
            connection.delete(schedule_key)

    @classmethod
    def get_schedules(cls, app=None):
        ensure_conf(app)
        connection = get_redis(app)
        schedules_keys = connection.scan_iter("redbeat:sc4snmp;*")
        schedule_objects = []
        for key in schedules_keys:
            schedule_obj = connection.hgetall(key)
            definition = cls.decode_definition(schedule_obj["definition"])
            meta = cls.decode_definition(schedule_obj["meta"])
            entry = cls(app=app, **definition)
            entry.last_run_at = meta['last_run_at']
            schedule_objects.append(entry)

        return schedule_objects

    @property
    def due_at(self):
        # never run => due now
        if self.last_run_at is None:
            return self._default_now()

        delta = self.schedule.remaining_estimate(self.last_run_at)
        # if no delta, means no more events after the last_run_at.
        if delta is None:
            return None

        # overdue => due now
        if delta.total_seconds() < 0:
            return self._default_now()

        return self.last_run_at + delta

    @property
    def key(self):
        return self.generate_key(app=self.app, name=self.name)

    @property
    def score(self):
        """return UTC based UNIX timestamp"""
        if self.due_at is None:
            # Scores < zero are ignored on each tick.
            return -1
        return to_timestamp(self.due_at)

    @property
    def rank(self):
        return get_redis(self.app).zrank(self.app.redbeat_conf.schedule_key, self.key)

    def save(self):
        definition = {
            'name': self.name,
            'task': self.task,
            'args': self.args,
            'kwargs': self.kwargs,
            'options': self.options,
            'schedule': self.schedule,
            'enabled': self.enabled,
            'run_immediately': self.run_immediately,
        }
        meta = {
            'last_run_at': self.last_run_at,
        }
        if self.first_run_kwargs:
            definition['first_run_kwargs'] = dict(self.first_run_kwargs)
        with get_redis(self.app).pipeline() as pipe:
            pipe.hset(self.key, 'definition', json.dumps(definition, cls=RedBeatJSONEncoder))
            pipe.hsetnx(self.key, 'meta', json.dumps(meta, cls=RedBeatJSONEncoder))
            pipe.zadd(self.app.redbeat_conf.schedule_key, {self.key: self.score})
            if self._strtobool(self.run_immediately):
                pipe.sadd("run_immediately", self.key)
            pipe.execute()

        return self

    def delete(self):
        with get_redis(self.app).pipeline() as pipe:
            pipe.zrem(self.app.redbeat_conf.schedule_key, self.key)
            pipe.delete(self.key)
            pipe.execute()

    def _check_policy(self, fingerprint):
        # Another writer may have changed the settings after this entry was read.
        # Ask the caller to reload rather than save using the older settings.
        if fingerprint != self.jitter_fingerprint:
            raise EntryPolicyChanged('The applied jitter policy changed')

    def _check_entry(self, entry_id):
        # Reject an older copy after deletion and creation under the same name.
        if self.decode_json(entry_id) != self.entry_id:
            raise KeyError(self.key)

    @_retry_watch_conflicts
    def save_guarded(self, *, create=False):
        """
        Save a loaded ordinary entry without overwriting a new managed policy.

        Like save(), this writes the caller's definition and due score while
        retaining existing metadata. It also checks the loaded identity and
        policy before writing, so a caller can reload after a policy change.
        Creation requires an absent hash and assigns a new identity. Updating
        a legacy entry without an identity does not add one.

        :param create: Whether to create an absent ordinary entry

        :return: Freshly loaded saved entry
        :raises KeyError: If the schedule is missing or its creation identity changed
        :raises ValueError: If the entry is managed or creation finds an existing hash
        :raises EntryPolicyChanged: If the saved entry acquired another policy
        """
        with get_redis(self.app).pipeline(transaction=True) as pipe:
            pipe.watch(self.key)
            definition = pipe.hget(self.key, 'definition')
            fingerprint = self.decode_json(pipe.hget(self.key, 'jitter_fingerprint'))
            entry_id = pipe.hget(self.key, 'entry_id')
            if create:
                if pipe.exists(self.key):
                    raise ValueError('Cannot replace an existing schedule hash')
                self.entry_id = str(uuid4())
            else:
                if not definition:
                    raise KeyError(self.key)
                self._check_entry(entry_id)
            # Beat may have enabled jitter while this update was being prepared.
            # Do not overwrite the new timing with the older copy.
            self._check_policy(fingerprint)
            if fingerprint:
                raise ValueError('Managed entries require their guarded update operation')
            definition = self._jitter_definition()
            if self.first_run_kwargs:
                definition['first_run_kwargs'] = dict(self.first_run_kwargs)
            meta = {'last_run_at': self.last_run_at}
            pipe.multi()
            pipe.hset(self.key, 'definition', json.dumps(definition, cls=RedBeatJSONEncoder))
            pipe.hsetnx(self.key, 'meta', json.dumps(meta, cls=RedBeatJSONEncoder))
            if create:
                pipe.hset(self.key, 'entry_id', json.dumps(self.entry_id))
            pipe.zadd(self.app.redbeat_conf.schedule_key, {self.key: self.score})
            if self._strtobool(self.run_immediately):
                pipe.sadd('run_immediately', self.key)
            pipe.execute()
        return type(self).from_key(self.key, app=self.app)

    def _jitter_definition(self):
        """
        Build the base definition for exact creation and payload updates.

        :return: Recurring definition without pending first-run arguments
        """
        return {
            'name': self.name,
            'task': self.task,
            'args': self.args,
            'kwargs': dict(self.kwargs),
            'options': self.options,
            'schedule': self.schedule,
            'enabled': self.enabled,
            'run_immediately': self.run_immediately,
        }

    def _validate_due_time(self, due):
        """
        Check an explicit interval deadline and normalize it to UTC.

        These operations represent the deadline as last_run_at plus one interval.
        Calendar and relative schedules cannot use that calculation. Metadata
        retains microseconds while the schedule index uses whole seconds.

        :param due: Requested timezone-aware datetime

        :return: The same instant in UTC, retaining microseconds
        :raises ValueError: If the interval or due time is unsupported
        """
        interval = getattr(self.schedule, 'run_every', None)
        if (
            not isinstance(interval, timedelta)
            or interval <= timedelta(0)
            or getattr(self.schedule, 'relative', False)
        ):
            raise ValueError('Explicit timing requires a positive non-relative interval')
        if not isinstance(due, datetime) or due.utcoffset() is None:
            raise ValueError('Due time must be a timezone-aware datetime')
        # Subtract intervals in UTC so a local daylight-saving change cannot
        # move the requested instant. The existing encoder keeps microseconds.
        return due.astimezone(timezone.utc)

    def save_with_first_due(self, first_due, fingerprint, *, first_run_kwargs=None):
        """
        Create a phased interval entry with its exact first due time.

        Definition, run metadata, due score, and fingerprint are saved together.
        First-run arguments override matching recurring kwargs in the first
        outgoing invocation. An existing hash is refused so creation cannot
        overwrite another entry or a previously accepted run. Use request_run()
        to change an existing managed entry's deadline.

        :param first_due: Timezone-aware first due time, normalized to UTC
        :param fingerprint: Nonempty identifier for the application's scheduling policy
        :param first_run_kwargs: Arguments overlaid on the first outgoing invocation.
            None uses this entry's pending arguments. An empty dict omits them.
            Recurring kwargs remain unchanged. The published message contains the
            overlaid arguments, so a task retry can reuse them.

        :return: This entry with its initial run metadata set
        :raises ValueError: If timing or fingerprint is invalid, or the entry exists
        """
        first_run_kwargs = dict(
            self.first_run_kwargs if first_run_kwargs is None else first_run_kwargs
        )
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError('A jitter fingerprint is required')
        first_due = self._validate_due_time(first_due)
        definition = self._jitter_definition()
        # Guarded writes compare this identity to reject older copies after
        # recreation, even when the name, timing and payload are unchanged.
        entry_id = str(uuid4())
        if first_run_kwargs:
            definition['first_run_kwargs'] = first_run_kwargs
        definition['run_immediately'] = 'False'
        # For a 60s interval due at 12:01:20, save last_run_at as 12:00:20.
        # This produces the chosen due without recording an actual execution.
        meta = {
            'last_run_at': first_due - self.schedule.run_every,
            'total_run_count': self.total_run_count,
        }
        with get_redis(self.app).pipeline(transaction=True) as pipe:
            pipe.watch(self.key)
            if pipe.exists(self.key):
                raise ValueError('Cannot replace an existing schedule hash')
            # Save the chosen first time and matching run information together.
            # Ordinary save() can calculate a different first time.
            pipe.multi()
            pipe.hset(self.key, 'definition', json.dumps(definition, cls=RedBeatJSONEncoder))
            pipe.hset(self.key, 'meta', json.dumps(meta, cls=RedBeatJSONEncoder))
            pipe.hset(self.key, 'jitter_fingerprint', json.dumps(fingerprint))
            pipe.hset(self.key, 'entry_id', json.dumps(entry_id))
            pipe.zadd(self.app.redbeat_conf.schedule_key, {self.key: to_timestamp(first_due)})
            # Wait for the chosen first time instead of running immediately.
            pipe.srem('run_immediately', self.key)
            pipe.execute()
        self.last_run_at = first_due - self.schedule.run_every
        self.run_immediately = 'False'
        self.jitter_fingerprint = fingerprint
        self.first_run_kwargs = first_run_kwargs
        self.entry_id = entry_id
        return self

    @_retry_watch_conflicts
    def request_run(self, due, *, run_immediately=False, fingerprint=None):
        """
        Request a managed run using the earliest pending request deadline.

        Repeated delayed requests keep the earliest due and latest payload.
        To request an urgent run, pass the current time as due. A later request
        cannot postpone it, though the latest request still supplies the payload.
        The existing immediate flag marks an accepted managed request. Its
        interval anchor stores the deadline, which is checked before dispatch.
        This temporary anchor is not an actual reservation timestamp. The run
        count is preserved.
        First-invocation arguments are cleared because this is a refresh.
        Identical deadlines and payloads may coalesce into one invocation.
        Redis write conflicts allow three total attempts, at most two retries.

        :param due: Timezone-aware requested due time, normalized to UTC
        :param run_immediately: Whether to prompt immediate selection even for a
            future deadline. Due or overdue requests always prompt selection.
            This flag does not bypass the due-time check.
        :param fingerprint: Applied policy after an intentional interval change

        :return: This entry with its accepted deadline and updated definition
        :raises KeyError: If the schedule is missing or its creation identity changed
        :raises ValueError: If timing, interval or applied policy is invalid
        """
        due = self._validate_due_time(due)
        run_immediately = self._strtobool(run_immediately)
        applied_fingerprint = self.jitter_fingerprint if fingerprint is None else fingerprint
        if not isinstance(applied_fingerprint, str) or not applied_fingerprint:
            raise ValueError('A jitter fingerprint is required')

        with get_redis(self.app).pipeline(transaction=True) as pipe:
            pipe.watch(self.key)
            raw_definition = pipe.hget(self.key, 'definition')
            current_fingerprint = self.decode_json(pipe.hget(self.key, 'jitter_fingerprint'))
            raw_meta = pipe.hget(self.key, 'meta')
            entry_id = pipe.hget(self.key, 'entry_id')
            if not raw_definition:
                raise KeyError(self.key)
            current = self.decode_definition(raw_definition)
            self._check_entry(entry_id)
            self._check_policy(current_fingerprint)
            if fingerprint is None and current['schedule'] != self.schedule:
                raise ValueError('Changing a request interval requires its applied policy')
            if not current_fingerprint:
                raise ValueError('A managed update requires an applied jitter policy')

            current_meta = self.decode_meta(raw_meta)
            if self._strtobool(current.get('run_immediately', 'False')):
                # If a refresh is already waiting, a later request must not delay it.
                # Read its due using the old interval before applying an interval edit.
                accepted_due = current_meta['last_run_at'] + current['schedule'].run_every
                due = min(due, accepted_due)
            definition = self._jitter_definition()
            # Mark this as a requested run and save when it should happen below.
            # Reuse the existing fields rather than create a separate request record.
            definition['run_immediately'] = 'True'
            meta = {
                'last_run_at': due - self.schedule.run_every,
                'total_run_count': current_meta.get('total_run_count', 0),
            }
            pipe.multi()
            pipe.hset(self.key, 'definition', json.dumps(definition, cls=RedBeatJSONEncoder))
            pipe.hset(self.key, 'meta', json.dumps(meta, cls=RedBeatJSONEncoder))
            if applied_fingerprint != current_fingerprint:
                pipe.hset(self.key, 'jitter_fingerprint', json.dumps(applied_fingerprint))
            pipe.zadd(self.app.redbeat_conf.schedule_key, {self.key: to_timestamp(due)})
            # Prompt selection for urgent or overdue requests. Every request
            # also retains its deadline in the schedule index for later selection.
            if run_immediately or due <= self._default_now():
                pipe.sadd('run_immediately', self.key)
            else:
                pipe.srem('run_immediately', self.key)
            pipe.execute()
        self.run_immediately = definition['run_immediately']
        self.first_run_kwargs = {}
        self.jitter_fingerprint = applied_fingerprint
        self.last_run_at = meta['last_run_at']
        self.total_run_count = meta['total_run_count']
        return self

    @_retry_watch_conflicts
    def update_definition(self):
        """
        Update a managed entry's payload without changing its saved timing.

        The immediate flag and pending first-invocation arguments remain as stored.
        Metadata, fingerprint, due score, and immediate-set membership are
        untouched. This operation does not request a run. A definition that
        serializes identically to the stored JSON does not need another write.
        Redis write conflicts allow three total attempts, at most two retries.

        :return: This entry after saving the updated definition
        :raises KeyError: If the schedule is missing or its creation identity changed
        :raises ValueError: If the interval or applied policy is incompatible
        """
        with get_redis(self.app).pipeline(transaction=True) as pipe:
            pipe.watch(self.key)
            raw_definition = pipe.hget(self.key, 'definition')
            fingerprint = self.decode_json(pipe.hget(self.key, 'jitter_fingerprint'))
            entry_id = pipe.hget(self.key, 'entry_id')
            if not raw_definition:
                raise KeyError(self.key)

            current = self.decode_definition(raw_definition)

            self._check_entry(entry_id)
            self._check_policy(fingerprint)
            if current['schedule'] != self.schedule:
                raise ValueError('A definition-only update cannot change the interval')
            if not fingerprint:
                raise ValueError('A managed update requires an applied jitter policy')

            first_run_kwargs = current.get('first_run_kwargs', {})
            definition = self._jitter_definition()
            # A refresh may have been requested while this update was prepared.
            # Keep that request and any arguments still waiting for their first run.
            if first_run_kwargs:
                definition['first_run_kwargs'] = dict(first_run_kwargs)
            definition['run_immediately'] = current.get('run_immediately', 'False')
            encoded_definition = json.dumps(definition, cls=RedBeatJSONEncoder)
            # Compare JSON text so byte responses work and argument type changes are kept.
            if encoded_definition != json.dumps(json.loads(raw_definition)):
                pipe.multi()
                pipe.hset(self.key, 'definition', encoded_definition)
                pipe.execute()
        self.kwargs = definition['kwargs']
        self.run_immediately = definition['run_immediately']
        self.first_run_kwargs = dict(first_run_kwargs)

        return self

    def set_enable(self, enabled):
        self.enabled = enabled

    def set_run_immediately(self, run_immediately):
        self.run_immediately = run_immediately

    def _next_instance(self, last_run_at=None, only_update_last_run_at=False):
        """
        Reserve one invocation without changing the stored recurring arguments.

        Entries with a policy marker or creation identity use guarded writes.
        These reject changes to the loaded policy or run state and check pending
        request payloads. An optional application policy selects the following
        deadline and policy marker. Its proposal is stored with this reservation.
        Unmarked entries without a proposal keep their existing write path.

        :param last_run_at: Reservation timestamp, or the application's current time
        :param only_update_last_run_at: Whether to retain the previous run count

        :return: Outgoing entry, with first-run arguments overlaid for guarded entries
        :raises KeyError: If a guarded entry is missing or its creation identity changed
        :raises ValueError: If compared guarded state changed before reservation
        """
        entry = super()._next_instance(last_run_at=last_run_at)

        if only_update_last_run_at:
            # rollback the update to total_run_count
            entry.total_run_count = self.total_run_count

        meta = {
            'last_run_at': entry.last_run_at,
            'total_run_count': entry.total_run_count,
        }
        policy = None if only_update_last_run_at else self.app.conf.get('redbeat_entry_policy')

        # Copy kwargs so the first-run overlay leaves the recurring dict unchanged.
        if self.jitter_fingerprint or self.entry_id:
            entry.kwargs = dict(entry.kwargs)
            entry.kwargs.update(self.first_run_kwargs)
            entry.first_run_kwargs = {}
            entry.run_immediately = 'False'
            self._save_phased_reservation(entry, meta, policy=policy)
            return entry

        # Preview policy adoption for an unmarked entry. A proposal changes the
        # following run while this invocation keeps its already-selected timing.
        if policy and self._following_timing(entry, policy) is not None:
            if self._save_phased_reservation(entry, meta, policy=policy):
                return entry

        # If no special timing update applies, save the next run as before.
        with get_redis(self.app).pipeline() as pipe:
            pipe.hset(self.key, 'meta', json.dumps(meta, cls=RedBeatJSONEncoder))
            pipe.zadd(self.app.redbeat_conf.schedule_key, {entry.key: entry.score})
            pipe.execute()

        return entry

    __next__ = next = _next_instance

    def _following_timing(self, entry, policy):
        """
        Ask the application for the next run's optional deadline and policy marker.

        None keeps ordinary recurrence. A pair selects an exact following due
        and fingerprint. A None due changes only the fingerprint, and a None
        fingerprint removes the applied policy. An exception or invalid proposal
        is ignored, so the reservation uses ordinary recurrence. Unmarked entries
        preview the policy before the guarded read. The callback must not change
        scheduling state because it runs again under WATCH and may repeat after
        a Redis write conflict.

        :param entry: Recurring definition with the prospective reservation time
        :param policy: Application callback that does not change scheduling state

        :return: Validated due and fingerprint pair, or None
        """
        try:
            proposal = policy(entry)
            if proposal is None:
                return None
            due, fingerprint = proposal
            if due is not None:
                due = self._validate_due_time(due)
                if due <= entry.last_run_at:
                    raise ValueError('The following due must be after its reservation')
            if fingerprint is not None and (not isinstance(fingerprint, str) or not fingerprint):
                raise ValueError('A nonempty policy fingerprint is required')
            return due, fingerprint
        except Exception:
            logger.exception('beat: Following-run policy failed for %s', self.name)
            return None

    @_retry_watch_conflicts
    def _save_phased_reservation(self, entry, meta, *, policy=None):
        """
        Save a phased reservation without replacing a newer task definition.

        Read the latest payload before clearing consumed first-run state. An
        unchanged definition is left untouched. A changed requested deadline
        or payload rejects this stale reservation so a later selection can use
        the newer request. Identical requests may coalesce.
        Redis write conflicts allow three total attempts, at most two retries,
        rereading the saved state each time.
        Deleted entries are not recreated. Changed policy, interval and run
        metadata are rejected. A stored creation identity also rejects a
        replacement under the same name, even when its other fields match.

        :param entry: Outgoing entry with this reservation's metadata
        :param meta: Run metadata to store for this reservation
        :param policy: Optional pure callback proposing the next deadline and policy

        :return: Whether a guarded reservation was stored
        :raises KeyError: If the schedule is missing or its creation identity changed
        :raises ValueError: If policy, interval, or run state is incompatible
        :raises redis.exceptions.WatchError: If concurrent writes prevent saving
        """
        with get_redis(self.app).pipeline(transaction=True) as pipe:
            pipe.watch(self.key)
            raw_definition = pipe.hget(self.key, 'definition')
            fingerprint = self.decode_json(pipe.hget(self.key, 'jitter_fingerprint'))
            raw_meta = pipe.hget(self.key, 'meta')
            entry_id = pipe.hget(self.key, 'entry_id')
            if not raw_definition:
                raise KeyError(self.key)

            # Another writer may have updated the task while Beat prepared this run.
            # Keep its newer task arguments instead of saving Beat's older copy.
            definition = self.decode_definition(raw_definition)
            self._check_entry(entry_id)
            self._check_policy(fingerprint)
            if definition['schedule'] != self.schedule or definition['task'] != self.task:
                raise ValueError('The schedule task or interval changed before reservation')

            current_meta = self.decode_meta(raw_meta)
            # Another reservation or refresh may have changed this run's timing.
            # Do not save this older selection over the newer run information.
            if (
                current_meta.get('last_run_at') != self.last_run_at
                or current_meta.get('total_run_count', 0) != self.total_run_count
            ):
                raise ValueError('The schedule run state changed before reservation')

            if (
                not fingerprint
                and not self.entry_id
                and pipe.sismember('run_immediately', self.key)
            ):
                # An ordinary immediate request is still pending. Let the ordinary
                # reservation path handle it before applying a following-run policy.
                return False

            if (fingerprint or self.entry_id) and self._strtobool(
                definition.get('run_immediately', 'False')
            ):
                # A newer refresh may carry different arguments from this selection.
                # Compare both copies in the same saved format before accepting it.
                requested_definition = self._jitter_definition()
                if self.first_run_kwargs:
                    requested_definition['first_run_kwargs'] = self.first_run_kwargs
                requested_definition = self.decode_definition(
                    json.dumps(requested_definition, cls=RedBeatJSONEncoder)
                )
                current_request = dict(definition)
                # reschedule() saves target, while ordinary save() omits it.
                # Compare it when the stored request includes that field.
                if 'target' in current_request:
                    requested_definition['target'] = self.target
                if not fingerprint:
                    # Ordinary is_due() clears its local immediate flag. The
                    # saved request still belongs to this run when its payload
                    # and run information match.
                    requested_definition['run_immediately'] = current_request['run_immediately']
                for flag in ('enabled', 'run_immediately'):
                    requested_definition[flag] = self._strtobool(requested_definition.get(flag))
                    current_request[flag] = self._strtobool(current_request.get(flag))
                if current_request != requested_definition:
                    raise ValueError('The requested payload changed before reservation')
            # Remove the first-run arguments used by this run. If another writer
            # saved different first-run arguments, leave them for their own run.
            definition_changed = False
            if self.first_run_kwargs and (
                definition.get('first_run_kwargs') == self.first_run_kwargs
            ):
                definition.pop('first_run_kwargs')
                definition_changed = True
            if definition.get('run_immediately') != 'False':
                definition['run_immediately'] = 'False'
                definition_changed = True

            proposal = None
            if policy and not definition.get('first_run_kwargs'):
                # Defer policy changes while newer first-run arguments remain.
                # Once consumed, the policy can choose the following deadline.
                following = type(self)(
                    app=self.app,
                    jitter_fingerprint=fingerprint,
                    entry_id=self.entry_id,
                    **definition,
                    **meta,
                )
                proposal = self._following_timing(following, policy)
            if not fingerprint and not self.entry_id and proposal is None:
                return False
            stored_meta = meta
            next_score = entry.score
            next_fingerprint = fingerprint
            if proposal is not None:
                due, next_fingerprint = proposal
                if due is not None:
                    # Keep the precise deadline in the interval anchor. Beat's
                    # whole-second index may select it before is_due() allows a run.
                    stored_meta = dict(meta, last_run_at=due - self.schedule.run_every)
                    next_score = to_timestamp(due)

            pipe.multi()
            pipe.hset(self.key, 'meta', json.dumps(stored_meta, cls=RedBeatJSONEncoder))
            # Rewrite the definition only when consuming first-run arguments or
            # normalizing the immediate flag, so ordinary runs avoid this write.
            if definition_changed:
                pipe.hset(self.key, 'definition', json.dumps(definition, cls=RedBeatJSONEncoder))
            if next_fingerprint != fingerprint:
                if next_fingerprint is None:
                    pipe.hdel(self.key, 'jitter_fingerprint')
                else:
                    pipe.hset(self.key, 'jitter_fingerprint', json.dumps(next_fingerprint))
            pipe.zadd(self.app.redbeat_conf.schedule_key, {entry.key: next_score})
            pipe.srem('run_immediately', self.key)
            pipe.execute()
        return True

    def reschedule(self, last_run_at=None):
        self.last_run_at = last_run_at or self._default_now()
        meta = {
            'last_run_at': self.last_run_at,
        }
        definition = {
            'name': self.name,
            'task': self.task,
            'args': self.args,
            'kwargs': self.kwargs,
            'options': self.options,
            'schedule': self.schedule,
            'target': self.target,
            'enabled': self.enabled,
            'run_immediately': self.run_immediately,
        }
        if self.first_run_kwargs:
            definition['first_run_kwargs'] = dict(self.first_run_kwargs)
        with get_redis(self.app).pipeline() as pipe:
            pipe.hset(self.key, 'meta', json.dumps(meta, cls=RedBeatJSONEncoder))
            pipe.hset(self.key, 'definition', json.dumps(definition, cls=RedBeatJSONEncoder))
            pipe.zadd(self.app.redbeat_conf.schedule_key, {self.key: self.score})
            if self._strtobool(self.run_immediately):
                pipe.sadd("run_immediately", self.key)
            pipe.execute()

    def is_due(self):
        if not self._strtobool(self.enabled):
            return False, 5.0  # 5 second delay for re-enable.

        if self.jitter_fingerprint and self._strtobool(self.run_immediately):
            # Managed requests use run_immediately as a pending-request marker,
            # including refreshes with a future deadline. Lookahead can select
            # them early, so check the saved deadline before allowing dispatch.
            # Urgent requests with a due-now deadline pass this check immediately.
            remaining = (
                self.last_run_at + self.schedule.run_every - self._default_now()
            ).total_seconds()
            return remaining <= 0, remaining if remaining > 0 else self.schedule.seconds

        if self._strtobool(self.run_immediately):
            self.run_immediately = False
            return True, self.schedule.seconds

        return self.schedule.is_due(
            self.last_run_at or datetime(MINYEAR, 1, 2, tzinfo=self.schedule.tz)
        )

    def _strtobool(self, v):
        return str(v).lower() in ("yes", "true", "t", "1")


class RedBeatScheduler(Scheduler):
    # how often should we sync in schedule information
    # from the backend redis database
    Entry = RedBeatSchedulerEntry

    lock = None

    #: The default lock timeout in seconds.
    lock_timeout = DEFAULT_MAX_INTERVAL * 5

    def __init__(self, app, lock_key=None, lock_timeout=None, **kwargs):
        ensure_conf(app)  # set app.redbeat_conf
        super(RedBeatScheduler, self).__init__(app, **kwargs)

        self.lock_key = lock_key or app.redbeat_conf.lock_key
        self.lock_timeout = (
            lock_timeout
            or app.redbeat_conf.lock_timeout
            or self.max_interval * 5
            or self.lock_timeout
        )

    def setup_schedule(self):
        # cleanup old static schedule entries
        client = get_redis(self.app)
        previous = {key for key in client.smembers(self.app.redbeat_conf.statics_key)}
        removed = previous.difference(self.app.redbeat_conf.schedule.keys())

        for name in removed:
            logger.debug("beat: Removing old static schedule entry '%s'.", name)
            with client.pipeline() as pipe:
                RedBeatSchedulerEntry(name, app=self.app).delete()
                pipe.srem(self.app.redbeat_conf.statics_key, name)
                pipe.execute()

        # setup static schedule entries
        self.install_default_entries(self.app.redbeat_conf.schedule)
        if self.app.redbeat_conf.schedule:
            self.update_from_dict(self.app.redbeat_conf.schedule)

            # keep track of static schedule entries,
            # so we notice when any are removed at next startup
            client.sadd(self.app.redbeat_conf.statics_key, *self.app.redbeat_conf.schedule.keys())

    def update_from_dict(self, dict_):
        for name, entry in dict_.items():
            redis_key = self.Entry.generate_key(self.app, name)

            try:
                redis_entry = self.Entry.from_key(redis_key, app=self.app)
                entry = self._maybe_entry(name, entry)
                entry.last_run_at = (
                    redis_entry.last_run_at
                )  # update definition while preserving last_run_at
            except KeyError:
                try:
                    entry = self._maybe_entry(name, entry)
                except Exception as exc:
                    logger.error(ADD_ENTRY_ERROR, name, exc, entry)
                    continue
            entry.save()  # store into redis
            logger.debug("beat: Stored entry: %s", entry)

    def reserve(self, entry):
        new_entry = next(entry)
        return new_entry

    @property
    def schedule(self):
        logger.debug('beat: Selecting tasks')

        max_due_at = to_timestamp(self.app.now())
        client = get_redis(self.app)

        with client.pipeline() as pipe:
            pipe.zrangebyscore(self.app.redbeat_conf.schedule_key, 0, max_due_at)

            # peek into the next tick to accuratly calculate sleep between ticks
            pipe.zrangebyscore(
                self.app.redbeat_conf.schedule_key,
                '({}'.format(max_due_at),
                max_due_at + self.max_interval,
                start=0,
                num=1,
            )
            pipe.smembers("run_immediately")
            pipe.delete("run_immediately")
            due_tasks, maybe_due, run_immediately_tasks, _ = pipe.execute()

        d = {}
        for key in list(run_immediately_tasks) + due_tasks + maybe_due:
            try:
                entry = self.Entry.from_key(key, app=self.app)
            except KeyError:
                logger.warning('beat: Failed to load %s, removing', key)
                client.zrem(self.app.redbeat_conf.schedule_key, key)
                continue

            d[entry.name] = entry

        return d

    def maybe_due(self, entry, **kwargs):
        is_due, next_time_to_run = entry.is_due()

        if is_due:
            logger.info('Scheduler: Sending due task %s (%s)', entry.name, entry.task)
            try:
                result = self.apply_async(entry, **kwargs)
            except Exception as exc:
                logger.exception('Scheduler: Message Error: %s', exc)
            else:
                if result and hasattr(result, 'id'):
                    logger.debug('Scheduler: %s sent. id->%s', entry.task, result.id)
                else:
                    logger.debug('Scheduler: %s sent.', entry.task)
        return next_time_to_run

    def tick(self, min=min, **kwargs):
        if self.lock_key:
            logger.debug('beat: Extending lock...')
            self.lock.extend(int(self.lock_timeout))

        remaining_times = []
        try:
            for entry in self.schedule.values():
                next_time_to_run = self.maybe_due(entry, **self._maybe_due_kwargs)
                if next_time_to_run:
                    remaining_times.append(next_time_to_run)
        except RuntimeError:
            logger.debug('beat: RuntimeError', exc_info=True)

        return min(remaining_times + [self.max_interval])

    def close(self):
        if self.lock:
            logger.info('beat: Releasing lock')
            if self.lock.owned():
                self.lock.release()
            self.lock = None
        super().close()

    @property
    def info(self):
        info = ['       . redis -> {}'.format(maybe_sanitize_url(self.app.redbeat_conf.redis_url))]
        if self.lock_key:
            info.append(
                '       . lock -> `{}` {} ({}s)'.format(
                    self.lock_key, humanize_seconds(self.lock_timeout), self.lock_timeout
                )
            )
        return '\n'.join(info)

    @cached_property
    def _maybe_due_kwargs(self):
        """handle rename of publisher to producer"""
        return {'producer': self.producer}


@beat_init.connect
def acquire_distributed_beat_lock(sender=None, **kwargs):
    """
    Attempt to acquire lock on startup

    Celery will squash any exceptions raised here. If one is raised
    scheduler.lock will be None while scheduler.lock_key is set
    """
    scheduler = sender.scheduler
    if not scheduler.lock_key:
        return

    logger.debug('beat: Acquiring lock...')
    redis_client = get_redis(scheduler.app)

    lock = redis_client.lock(
        scheduler.lock_key,
        timeout=scheduler.lock_timeout,
        sleep=scheduler.max_interval,
    )
    # overwrite redis-py's extend script
    # which will add additional timeout instead of extend to a new timeout
    lock.lua_extend = redis_client.register_script(LUA_EXTEND_TO_SCRIPT)
    lock.acquire()
    logger.info('beat: Acquired lock')
    scheduler.lock = lock
