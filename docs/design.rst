
Design
------
At its core RedBeat uses a Sorted Set to store the schedule as a priority queue.
Additionally, task details are stored within a hash key mapping to the task definition and metadata.

The sortted schedule set contains task keys sorted by the next scheduled run time, as time since epoch.

For each tick of Beat

  #. check if it still owns the lock, if not, exit with ``LockNotOwnedError``
  #. get list of due keys and due next tick
  #. retrieve definitions and metadata for all keys from previous step
  #. update task metadata and reschedule with next run time for task
  #. call due tasks using async_apply, this enqueues tasks for workers to run
  #. calculate time to sleep until start of next tick using due next tick tasks

Scheduling
~~~~~~~~~~~~
Assuming your `redbeat_key_prefix` config values is set to `'redbeat:'`
(default) you will also need to insert the new task into the schedule with::

    zadd redbeat::schedule 0 new-task-name

The score is the next time the task should run formatted as a UNIX timestamp.

Metadata
~~~~~~~~~~~
Applications may also want to manipulate the task metadata to have more control over when a task runs.
The meta key contains a JSON blob as follows::

    {
        'last_run_at': {
            '__type__': 'datetime',
            'year': 2015,
            'month': 12,
            'day': 29,
            'hour': 16,
            'minute': 45,
            'microsecond': 231
        },
        'total_run_count'; 23
    }

For instance by default ```last_run_at``` corresponds to when Beat dispatched the task, but depending on queue latency it might not run immediately, but the application could update the metadata with
the actual run time, allowing intervals to be relative to last execution rather than last dispatch.

High Availability
~~~~~~~~~~~~~~~~~
Redbeat use a lock in redis to prevent multiple node running.
You can safely start multiple nodes as backup, when the running node fails or
experience network problems, after ``redbeat_lock_timeout`` seconds,
another node will acquire the lock and start running.

When the previous node back online, it will notice that itself no longer holds
the lock and exit with an Exception(Would be better if you use systemd or supervisord
to restart it as a backup node).


Timezone
~~~~~~~~~~~
To support non-UTC timezones it's useful to be able to reason about which values are localized and which are always UTC.

Regardless, of what timezone is used, the ```redbeat_key_prefix:schedule``` sorted set is sorted by the UNIX timestamp, of time since the UNIX epoch in UTC.

Every task schedule maybe created with a timezone independent of the host timezone.
As a result most datetime objects need to be timezone aware.

    last_run_at, aware datetime
    now(), returns aware datetime
    due_at, returns aware datetime
    to_timestamp(), accepts aware datetime, returns seconds since epoch (UTC)
    score(), returns seconds since epoch for due_at

Explicit interval deadlines
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The opt-in ``save_with_first_due()`` and ``request_run()`` methods accept
timezone-aware deadlines. RedBeat converts them to UTC before calculating the
matching interval anchor. Fractional seconds in the deadline and interval are
preserved in JSON metadata. These methods require a positive, non-relative
interval schedule. Calendar schedules and relative rounding do not use this
anchor calculation. Other scheduling APIs keep their existing behavior.

The optional ``redbeat_entry_policy`` callback can select a following deadline
under the same rules. It does not move the current saved invocation. Applications
own their policy, including any jitter window, identity hashing, whole-second
rounding or exceptions for urgent work. RedBeat stores the supplied timing and
protects managed writes against concurrent updates and schedule replacement.

Redis lookup scores retain their existing whole-second precision. A candidate
can therefore be selected before a fractional deadline. The full-precision due
check still prevents early publication. Beat wake-up frequency, broker latency
and worker capacity can delay dispatch or execution. Subsecond timing is not a
real-time execution guarantee.
