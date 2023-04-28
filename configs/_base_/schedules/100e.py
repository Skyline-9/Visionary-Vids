# runtime settings

from optimizers.lion import Lion

stages = dict(
    epochs=100,
    optimizer=dict(type="Lion", lr=1e-4, weight_decay=1e-4),
    validation=dict(interval=1),
)
