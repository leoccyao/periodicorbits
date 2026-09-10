from typing import Iterable
from .. import trainer

class Train_Target(object):
    """Base generator target; the default never terminates training early."""
    def __init__(self):
        pass

    def __call__(self) -> 'trainer.Training_Checker':
        """Yield a result after each received ``(progress, trainer)`` pair."""
        progress, trainer = (yield)
        while True:
            result = False
            progress, trainer = (yield result)

class Not_Target(Train_Target):
    """A negation target that returns the opposite of what the target returns."""
    def __init__(self, target: Train_Target):
        super().__init__()
        self.target = target()
    

    def __call__(self) -> 'trainer.Training_Checker':
        next(self.target)

        progress, trainer = (yield)
        while True:
            result = not (self.target.send((progress, trainer)))
            progress, trainer = (yield result)


class Any_Target(Train_Target):
    """A union target that returns when any of the targets return."""
    def __init__(self, targets: Iterable[Train_Target]):
        super().__init__()
        self.targets = [target() for target in targets]


    def __call__(self) -> 'trainer.Training_Checker':
        for target in self.targets:
            next(target)
        
        progress, trainer = (yield)
        while True:
            result = any(target.send((progress, trainer)) for target in self.targets)
            progress, trainer = (yield result)


class All_Target(Train_Target):
    """An intersection target that succeeds when all component targets do."""
    def __init__(self, targets: Iterable[Train_Target]):
        super().__init__()
        self.targets = [target() for target in targets]


    def __call__(self) -> 'trainer.Training_Checker':
        for target in self.targets:
            next(target)
        
        progress, trainer = (yield)
        while True:
            result = all(target.send((progress, trainer)) for target in self.targets)
            progress, trainer = (yield result)

class Val_Target(Train_Target):
    """Succeed when validation loss is below ``val_target``."""
    def __init__(self, val_target=1e-10):
        super().__init__()
        self.val_target = val_target
    
    def __call__(self) -> 'trainer.Training_Checker':
        progress, trainer = (yield)
        while True:
            result = progress.val_loss < self.val_target
            progress, trainer = (yield result)


class With_Val_Limit(Any_Target):
    """Succeed when the wrapped target or a validation floor is reached."""
    def __init__(self, target: Train_Target, val_limit=1e-60):
        super().__init__([target, Val_Target(val_limit)])


class With_Val_Requirement(All_Target):
    """Require both the wrapped target and a validation threshold."""
    def __init__(self, target: Train_Target, val_limit=1):
            super().__init__([target, Val_Target(val_limit)])


class Val_Change_Factor(Train_Target):
    """Succeed when the ratio to the previous validation loss exceeds ``factor``."""
    def __init__(self, factor=1.0):
        super().__init__()
        self.factor = factor
    
    def __call__(self) -> 'trainer.Training_Checker':
        prev_val = float('inf')

        progress, trainer = (yield)
        while True:
            val = progress.val_loss
            result = val / prev_val > self.factor
            progress, trainer = (yield result)
            prev_val = val


class Val_Decrease_Epochs(Train_Target):
    """Succeed after validation loss has not improved for ``epochs``."""
    def __init__(self, epochs=1000):
        super().__init__()
        self.epochs = epochs
    
    def __call__(self) -> 'trainer.Training_Checker':
        prev_val = float('inf')
        prev_epoch = 0

        progress, trainer = (yield)
        while True:
            epoch, val = progress.epoch, progress.val_loss
            if val < prev_val:
                prev_val = val
                prev_epoch = epoch
            result = (epoch >= prev_epoch + self.epochs)
            progress, trainer = (yield result)         
