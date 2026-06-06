import unittest

from toyvllm.scheduler import Scheduler
from toyvllm.sequence import FinishReason, SequenceStatus


class SchedulerTest(unittest.TestCase):
    def test_fifo_admission_and_slot_reuse(self) -> None:
        scheduler = Scheduler(max_num_seqs=2)
        first = scheduler.add_request(
            [1],
            max_new_tokens=1,
            eos_token_ids=set(),
        )
        second = scheduler.add_request(
            [2],
            max_new_tokens=3,
            eos_token_ids=set(),
        )
        third = scheduler.add_request(
            [3],
            max_new_tokens=2,
            eos_token_ids=set(),
        )

        admitted = scheduler.admit_waiting(step=0)
        self.assertEqual([sequence.request_id for sequence in admitted], [0, 1])
        self.assertEqual(third.status, SequenceStatus.WAITING)

        reason = scheduler.append_token(first, 10, step=0)
        self.assertEqual(reason, FinishReason.LENGTH)
        self.assertEqual(first.status, SequenceStatus.FINISHED)

        admitted = scheduler.admit_waiting(step=0)
        self.assertEqual([sequence.request_id for sequence in admitted], [2])
        self.assertEqual(third.status, SequenceStatus.RUNNING)
        self.assertEqual(third.admitted_step, 0)
        self.assertEqual(
            [sequence.request_id for sequence in scheduler.running],
            [1, 2],
        )

    def test_eos_finishes_before_length_limit(self) -> None:
        scheduler = Scheduler(max_num_seqs=1)
        sequence = scheduler.add_request(
            [1],
            max_new_tokens=10,
            eos_token_ids={99},
        )
        scheduler.admit_waiting(step=0)
        reason = scheduler.append_token(sequence, 99, step=1)
        self.assertEqual(reason, FinishReason.EOS)
        self.assertEqual(sequence.finished_step, 1)
        self.assertTrue(scheduler.is_done)

    def test_admission_can_be_limited_to_one_request(self) -> None:
        scheduler = Scheduler(max_num_seqs=3)
        for token in range(3):
            scheduler.add_request(
                [token],
                max_new_tokens=1,
                eos_token_ids=set(),
            )
        admitted = scheduler.admit_waiting(step=0, max_sequences=1)
        self.assertEqual(len(admitted), 1)
        self.assertEqual(len(scheduler.waiting), 2)


if __name__ == "__main__":
    unittest.main()
