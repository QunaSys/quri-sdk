# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import pytest

from quri_parts.core.utils.recording import (
    DEBUG,
    INFO,
    RecordEntry,
    Recorder,
    RecordSession,
    recordable,
)


@recordable
def func_to_record(recorder: Recorder, x: int) -> int:
    recorder.info("x", x)
    recorder.info("2x", 2 * x)
    return 2 * x


def test_recordable() -> None:
    fid = func_to_record.id
    session = RecordSession()
    session.set_level(INFO, func_to_record)

    with session.start():
        assert func_to_record(3) == 6
        assert func_to_record(4) == 8

    records = session.get_records()
    history = list(records.get_history(func_to_record))
    assert len(history) == 2
    group0, group1 = history

    assert group0.func_id == fid
    assert group0.entries == [
        RecordEntry(INFO, fid, ("x", 3)),
        RecordEntry(INFO, fid, ("2x", 6)),
    ]

    assert group1.func_id == fid
    assert group1.entries == [
        RecordEntry(INFO, fid, ("x", 4)),
        RecordEntry(INFO, fid, ("2x", 8)),
    ]


def test_nested_sessions() -> None:
    fid = func_to_record.id
    session1 = RecordSession()
    session1.set_level(INFO, func_to_record)
    session2 = RecordSession()
    session2.set_level(INFO, func_to_record)

    with session1.start():
        assert func_to_record(1) == 2
        with session2.start():
            assert func_to_record(2) == 4
        assert func_to_record(3) == 6

    records1 = session1.get_records()
    history1 = list(records1.get_history(func_to_record))
    assert len(history1) == 3
    group0, group1, group2 = history1

    assert all(group.func_id == fid for group in history1)
    assert group0.entries == [
        RecordEntry(INFO, fid, ("x", 1)),
        RecordEntry(INFO, fid, ("2x", 2)),
    ]
    assert group1.entries == [
        RecordEntry(INFO, fid, ("x", 2)),
        RecordEntry(INFO, fid, ("2x", 4)),
    ]
    assert group2.entries == [
        RecordEntry(INFO, fid, ("x", 3)),
        RecordEntry(INFO, fid, ("2x", 6)),
    ]

    records2 = session2.get_records()
    history2 = list(records2.get_history(func_to_record))
    assert len(history2) == 1
    (group,) = history2

    assert group.func_id == fid
    assert group.entries == [
        RecordEntry(INFO, fid, ("x", 2)),
        RecordEntry(INFO, fid, ("2x", 4)),
    ]


@recordable
def nested_func(recorder: Recorder, x: int) -> int:
    recorder.info("x", x)
    if x // 2 == 1:
        y = nested_func(2 * x)
        recorder.info("y", y)
    else:
        y = x
    recorder.info("2y", 2 * y)
    return 2 * y


def test_nested_func_record() -> None:
    fid = nested_func.id
    session = RecordSession()
    session.set_level(INFO, nested_func)

    with session.start():
        assert nested_func(3) == 24
        assert nested_func(4) == 8

    records = session.get_records()
    history = list(records.get_history(nested_func))
    assert len(history) == 3
    group0, group1, group2 = history

    assert group0.func_id == fid
    assert group0.entries == [
        RecordEntry(INFO, fid, ("x", 3)),
        RecordEntry(INFO, fid, ("y", 12)),
        RecordEntry(INFO, fid, ("2y", 24)),
    ]

    assert group1.func_id == fid
    assert group1.entries == [
        RecordEntry(INFO, fid, ("x", 6)),
        RecordEntry(INFO, fid, ("2y", 12)),
    ]

    assert group2.func_id == fid
    assert group2.entries == [
        RecordEntry(INFO, fid, ("x", 4)),
        RecordEntry(INFO, fid, ("2y", 8)),
    ]


@recordable
def rec_level_func(recorder: Recorder, x: int) -> int:
    recorder.info("info", x)
    recorder.debug("debug", 2 * x)
    return 2 * x


class TestRecordLevel:
    def test_default_level(self) -> None:
        session = RecordSession()

        with session.start():
            assert rec_level_func(3) == 6

        records = session.get_records()
        history = list(records.get_history(rec_level_func))
        assert len(history) == 0

    def test_info_level(self) -> None:
        fid = rec_level_func.id
        session = RecordSession()
        session.set_level(INFO, rec_level_func)

        with session.start():
            assert rec_level_func(3) == 6

        records = session.get_records()
        history = list(records.get_history(rec_level_func))
        assert len(history) == 1
        (group,) = history

        assert group.entries == [RecordEntry(INFO, fid, ("info", 3))]

    def test_debug_level(self) -> None:
        fid = rec_level_func.id
        session = RecordSession()
        session.set_level(DEBUG, rec_level_func)

        with session.start():
            assert rec_level_func(3) == 6

        records = session.get_records()
        history = list(records.get_history(rec_level_func))
        assert len(history) == 1
        (group,) = history

        assert group.entries == [
            RecordEntry(INFO, fid, ("info", 3)),
            RecordEntry(DEBUG, fid, ("debug", 6)),
        ]


@recordable
def is_enabled_for_func(recorder: Recorder, x: int) -> int:
    # arg of is_enabled for (DEBUG) and recorder.info are inconsistent
    # for testing purpose
    if recorder.is_enabled_for(DEBUG):
        recorder.info("x", x)
    return 2 * x


class TestIsEnabledFor:
    def test_is_enabled_for_true(self) -> None:
        fid = is_enabled_for_func.id
        session = RecordSession()
        session.set_level(DEBUG, is_enabled_for_func)

        with session.start():
            assert is_enabled_for_func(3) == 6

        records = session.get_records()
        history = list(records.get_history(is_enabled_for_func))
        assert len(history) == 1
        (group,) = history

        assert group.entries == [RecordEntry(INFO, fid, ("x", 3))]

    def test_is_enabled_for_false(self) -> None:
        session = RecordSession()
        session.set_level(INFO, is_enabled_for_func)

        with session.start():
            assert is_enabled_for_func(3) == 6

        records = session.get_records()
        history = list(records.get_history(is_enabled_for_func))
        assert len(history) == 0

    def test_is_enabled_for_multiple_sessions(self) -> None:
        fid = is_enabled_for_func.id
        # session1 is set to the default level
        session1 = RecordSession()
        session2 = RecordSession()
        session2.set_level(DEBUG, is_enabled_for_func)

        with session1.start():
            with session2.start():
                assert is_enabled_for_func(3) == 6

        records1 = session1.get_records()
        history1 = list(records1.get_history(is_enabled_for_func))
        assert len(history1) == 0

        records2 = session2.get_records()
        history2 = list(records2.get_history(is_enabled_for_func))
        assert len(history2) == 1
        (group,) = history2

        assert group.entries == [RecordEntry(INFO, fid, ("x", 3))]


@recordable
def logging_func(recorder: Recorder, x: int) -> int:
    recorder.info("x", x)
    recorder.debug("2x", 2 * x)
    return 2 * x


class TestLogging:
    def test_default_logger_info(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO)

        session = RecordSession()
        session.set_level(INFO, logging_func)
        session.add_logger()

        with session.start():
            assert logging_func(3) == 6

        assert len(caplog.records) == 1
        log_record = caplog.records[0]
        assert log_record.name == f"root.quri_parts_recording.{logging_func.id.module}"
        assert log_record.levelno == logging.INFO
        assert log_record.message == f"{logging_func.id.qualname}: x=3"
        assert isinstance(getattr(log_record, "record_group"), int)

    def test_default_logger_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.DEBUG)

        session = RecordSession()
        session.set_level(DEBUG, logging_func)
        session.add_logger()

        with session.start():
            assert logging_func(3) == 6

        assert len(caplog.records) == 2
        log_record0, log_record1 = caplog.records

        assert log_record0.name == f"root.quri_parts_recording.{logging_func.id.module}"
        assert log_record0.levelno == logging.INFO
        assert log_record0.message == f"{logging_func.id.qualname}: x=3"
        assert isinstance(getattr(log_record0, "record_group"), int)

        assert log_record1.name == f"root.quri_parts_recording.{logging_func.id.module}"
        assert log_record1.levelno == logging.DEBUG
        assert log_record1.message == f"{logging_func.id.qualname}: 2x=6"
        assert isinstance(getattr(log_record1, "record_group"), int)

        assert getattr(log_record0, "record_group") == getattr(
            log_record1, "record_group"
        )

    def test_custom_logger(self, caplog: pytest.LogCaptureFixture) -> None:
        log_name = "test_recording"
        caplog.set_level(logging.INFO, logger=log_name)
        logger = logging.getLogger(log_name)

        session = RecordSession()
        session.set_level(INFO, logging_func)
        session.add_logger(logger)

        with session.start():
            assert logging_func(3) == 6

        assert len(caplog.records) == 1
        log_record = caplog.records[0]
        assert log_record.name == f"{log_name}.{logging_func.id.module}"
        assert log_record.levelno == logging.INFO
        assert log_record.message == f"{logging_func.id.qualname}: x=3"
        assert isinstance(getattr(log_record, "record_group"), int)

    def test_no_logging_by_default(self, caplog: pytest.LogCaptureFixture) -> None:
        session = RecordSession()
        session.set_level(INFO, logging_func)

        with session.start():
            assert logging_func(3) == 6

        assert len(caplog.record_tuples) == 0
