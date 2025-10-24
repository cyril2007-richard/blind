import unittest
import time
from audio.announcement_queue import AnnouncementQueue
from config.priority_rules import Priority

class TestAnnouncementQueue(unittest.TestCase):

    def setUp(self):
        self.queue = AnnouncementQueue()

    def test_add_announcement(self):
        self.assertTrue(self.queue.add_announcement("Hello", Priority.MEDIUM))
        self.assertEqual(self.queue.get_queue_size(), 1)

    def test_add_duplicate_announcement(self):
        self.assertTrue(self.queue.add_announcement("Hello", Priority.MEDIUM))
        self.assertFalse(self.queue.add_announcement("Hello", Priority.MEDIUM))
        self.assertEqual(self.queue.get_queue_size(), 1)

    def test_add_different_announcements(self):
        self.assertTrue(self.queue.add_announcement("Hello", Priority.MEDIUM))
        self.assertTrue(self.queue.add_announcement("World", Priority.MEDIUM))
        self.assertEqual(self.queue.get_queue_size(), 2)

    def test_interrupt_clears_queue(self):
        self.queue.add_announcement("Hello", Priority.MEDIUM)
        self.queue.add_announcement("World", Priority.HIGH, interrupt=True)
        self.assertEqual(self.queue.get_queue_size(), 1)

if __name__ == '__main__':
    unittest.main()
