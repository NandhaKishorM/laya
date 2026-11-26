"""Offline numeric feature contracts; no checkpoints, GPUs, or tokenizer downloads.

Run: python tests/test_numeric.py
"""
import dataclasses
from decimal import Decimal
import os
import random
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.numeric import AlignedDigit, DigitSlot, align_slots, decimal_slots  # noqa: E402


class NumericTests(unittest.TestCase):
    def test_scientific_places_and_characters(self):
        slots = decimal_slots('-0012.30e+2', start=5)
        self.assertEqual([s.digit for s in slots], [0, 0, 1, 2, 3, 0])
        self.assertEqual([s.power for s in slots], [5, 4, 3, 2, 1, 0])
        self.assertEqual([s.character for s in slots], [6, 7, 8, 9, 11, 12])
        self.assertTrue(all(s.negative for s in slots))

    def test_signed_zero_and_negative_exponent(self):
        self.assertEqual(decimal_slots('-0.00'),
                         (DigitSlot(0, 0, 1, True), DigitSlot(0, -1, 3, True), DigitSlot(0, -2, 4, True)))
        self.assertEqual([s.power for s in decimal_slots('+12E-3')], [-2, -3])

    def test_exact_decimal_reconstruction(self):
        rng = random.Random(41)
        literals = ['9007199254740993', '-9007199254740993', '0', '001.2300']
        literals += ['%s%d.%03de%+d' % ('-' if rng.randrange(2) else '+', rng.randrange(100000),
                                      rng.randrange(1000), rng.randrange(-8, 9)) for _ in range(100)]
        for literal in literals:
            with self.subTest(literal=literal):
                slots = decimal_slots(literal)
                reconstructed = sum(Decimal(s.digit).scaleb(s.power) for s in slots)
                if slots[0].negative:
                    reconstructed = -reconstructed
                self.assertEqual(reconstructed, Decimal(literal))

    def test_large_exponent_does_not_allocate_padding(self):
        slots = decimal_slots('12e1000000')
        self.assertEqual(len(slots), 2)
        self.assertEqual([s.power for s in slots], [1000001, 1000000])

    def test_reject_unsupported_literals_and_start(self):
        for value in ('', ' 12', '12 ', '12\n', '1,000', 'NaN', 'Infinity', '0xff', '.5', '1.',
                      '1e', '1e+', '1_000', '١٢', '１２', '--1', None, 12):
            with self.subTest(value=value), self.assertRaises(ValueError):
                decimal_slots(value)
        for start in (-1, True, 1.5, '1'):
            with self.assertRaises(ValueError):
                decimal_slots('12', start)

    def test_multi_digit_tokens_and_special_tokens(self):
        result = align_slots('x=123.40', [(2, 8)], [(0, 0), (0, 2), (2, 5), (5, 6), (6, 8), (0, 0)])
        self.assertEqual(result, (AlignedDigit(0, 2, 1, 2, False), AlignedDigit(0, 2, 2, 1, False),
                                  AlignedDigit(0, 2, 3, 0, False), AlignedDigit(0, 4, 4, -1, False),
                                  AlignedDigit(0, 4, 0, -2, False)))

    def test_unicode_character_offsets_and_quantity_order(self):
        text = 'π=12; β=-3e2'
        result = align_slots(text, [(8, 12), (2, 4)], [(0, 2), (2, 4), (4, 8), (8, 12)])
        self.assertEqual(result, (AlignedDigit(0, 3, 3, 2, True),
                                  AlignedDigit(1, 1, 1, 1, False), AlignedDigit(1, 1, 2, 0, False)))

    def test_each_literal_character_must_be_covered(self):
        text = '-12.5e+2'
        for missing in range(len(text)):
            offsets = [(i, i+1) for i in range(len(text)) if i != missing]
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                align_slots(text, [(0, len(text))], offsets)

    def test_invalid_offsets_and_overlaps(self):
        for offsets in ([(-1, 1)], [(0, 3)], [(1, 0)], [(False, 2)], [(0, 1.5)], [(0,)], ['01'],
                        [(0, 2), (1, 2)], None, '01'):
            with self.subTest(offsets=offsets), self.assertRaises(ValueError):
                align_slots('12', [(0, 2)], offsets)
        for spans in ([(0, 0)], [(0, 3)], [(0, 2), (1, 2)], [(0, True)], None, '01'):
            with self.subTest(spans=spans), self.assertRaises(ValueError):
                align_slots('12', spans, [(0, 2)])
        with self.assertRaises(ValueError):
            align_slots(None, [], [])

    def test_empty_inputs_and_no_implicit_number_extraction(self):
        self.assertEqual(align_slots('', [], []), ())
        self.assertEqual(align_slots('value=12', [], [(0, 8)]), ())
        with self.assertRaises(ValueError):
            align_slots('12', [(0, 2)], [])

    def test_records_are_immutable(self):
        for record in (decimal_slots('1')[0], align_slots('1', [(0, 1)], [(0, 1)])[0]):
            with self.assertRaises(dataclasses.FrozenInstanceError):
                record.digit = 2

    def test_import_has_no_ml_dependency(self):
        code = ('import sys; from laya.numeric import decimal_slots; '
                'assert decimal_slots("12")[0].power == 1; '
                'assert not {"torch", "transformers", "numpy"}.intersection(sys.modules)')
        subprocess.run([sys.executable, '-c', code], check=True, cwd=os.path.dirname(os.path.dirname(__file__)))


if __name__ == '__main__':
    unittest.main()
