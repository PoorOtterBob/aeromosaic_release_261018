#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase F1 + gradient checkpointing wrapper for non-YRD regions."""

from train_balanced import *
import train_balanced as _bal
import train_yrd_graph_cls_v2_rootregion_best_dagresidual as _base

_orig_build = _base.build_training_args


def _build_gc(args):
    ta = _orig_build(args)
    ta.gradient_checkpointing = True
    return ta


_base.build_training_args = _build_gc

_cls = _base.RawTsPrefixSequenceClassifier


def _gc_enable(self, gradient_checkpointing_kwargs=None):
    self.lm.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)


def _gc_disable(self):
    self.lm.gradient_checkpointing_disable()


_cls.gradient_checkpointing_enable = _gc_enable
_cls.gradient_checkpointing_disable = _gc_disable

if __name__ == "__main__":
    _bal.main_balanced()
