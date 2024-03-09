from typing import Callable
import tensorflow as tf
import tensorflow_addons as tfa


class ORLearningRateScheduler:
    """
    A custom learning rate scheduler that supports various scheduling strategies,
    including cosine decay with warmup, cyclical learning rate, cosine decay restarts,
    and plain cosine decay without warmup.

    Parameters
    ----------
    strategy : str
        The learning rate scheduling strategy. Supported strategies: 'cosine_decay_with_warmup',
        'cyclical', 'cosine_decay_restarts', 'cosine_decay_no_warmup'.
    steps_per_epoch : int
        The number of batches per epoch.
    total_epochs : int
        The total number of epochs for training.
    warmup_epochs : Optional[int], default=None
        The number of epochs to linearly ramp up the learning rate. Applicable for strategies
        that include a warmup phase.
    initial_lr : float, default=1e-5
        The initial learning rate.
    max_lr : float, default=1e-2
        The maximum learning rate, used in cyclical learning rate strategies.
    min_lr : float, default=1e-5
        The minimum or final learning rate after decay.
    decay_step_factor : float, default=0.9
        A factor to determine the total decay steps as a proportion of total training steps.
    t_mul : float, default=1.0
        A factor used in cosine decay restarts to multiply the period of each restart.
    m_mul : float, default=0.4
        A factor used in cosine decay restarts to multiply the amplitude of each restart.
    warmup_target : Optional[float], default=None
        The target learning rate at the end of the warmup phase. If not specified, `max_lr` is used.
    name : Optional[str], default=None
        An optional name for the scheduling strategy. If not specified, `strategy` is used.

    Attributes
    ----------
    scheduler : Callable
        The configured learning rate scheduler function or object.

    Methods
    -------
    get() -> Callable
        Returns the configured learning rate scheduler.
    
    Usage
    -----
    Here's how to use the `CustomLearningRateScheduler` in a TensorFlow training workflow:

    ```python
    

    # Configuration for the learning rate scheduler
    strategy = 'cosine_decay_with_warmup'
    steps_per_epoch = 100  # Assuming 100 batches per epoch
    total_epochs = 20
    warmup_epochs = 5
    initial_lr = 1e-5
    max_lr = 1e-2
    min_lr = 1e-5

    # Instantiate the custom learning rate scheduler
    lr_scheduler = CustomLearningRateScheduler(strategy=strategy,
                                               steps_per_epoch=steps_per_epoch,
                                               total_epochs=total_epochs,
                                               warmup_epochs=warmup_epochs,
                                               initial_lr=initial_lr,
                                               max_lr=max_lr,
                                               min_lr=min_lr)

    # Compile the model with the optimizer and learning rate scheduler
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_scheduler.get()),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Prepare your data (x_train, y_train) and train the model
    # model.fit(x_train, y_train, epochs=total_epochs)
    ```

    Note: Replace `'cosine_decay_with_warmup'` with any supported strategy you wish to use and adjust
    `steps_per_epoch`, `total_epochs`, `warmup_epochs`, `initial_lr`, `max_lr`, and `min_lr` according
    to your training configuration and data.

    """

    def __init__(
        self,
        strategy: str,
        steps_per_epoch: int,
        total_epochs: int,
        warmup_epochs: int = None,
        initial_lr: float = 1e-5,
        max_lr: float = 1e-2,
        min_lr: float = 1e-5,
        decay_step_factor: float = 0.9,
        t_mul: float = 1.0,
        m_mul: float = 0.4,
        warmup_target: float = None,
        name: str = None,
    ):
        
        self.strategy = strategy
        self.steps_per_epoch = steps_per_epoch
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_target = warmup_target if warmup_target else max_lr
        self.decay_step_factor = decay_step_factor
        self.t_mul = t_mul
        self.m_mul = m_mul
        self.name = name if name else strategy

        if strategy == "cosine_decay_with_warmup":
            self.scheduler = self._cosine_decay_with_warmup()
        elif strategy == "cyclical":
            self.scheduler = self._cyclical_lr()
        elif strategy == "cosine_decay_restarts":
            self.scheduler = self._cosine_decay_restarts()
        elif strategy == "cosine_decay_no_warmup":
            self.scheduler = self._cosine_decay_no_warmup()
        else:
            raise ValueError("Unsupported strategy specified.")

    def _cosine_decay_no_warmup(self)-> tf.keras.optimizers.schedules.LearningRateSchedule:

        decay_steps = (self.total_epochs * self.steps_per_epoch) * self.decay_step_factor
        alpha = self.min_lr / self.max_lr

        return tf.keras.experimental.CosineDecay(
            initial_learning_rate=self.initial_lr, decay_steps=decay_steps, alpha=alpha
        )

    def _cosine_decay_with_warmup(self)-> tf.keras.optimizers.schedules.LearningRateSchedule:

        assert self.warmup_epochs < self.total_epochs, "Warmup epochs must be less than total epochs."
        total_steps = self.total_epochs * self.steps_per_epoch
        warmup_steps = self.warmup_epochs * self.steps_per_epoch
        decay_steps = (total_steps - warmup_steps) * self.decay_step_factor
        alpha = self.min_lr / self.warmup_target

        return tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=self.initial_lr,
            decay_steps=decay_steps,
            warmup_target=self.warmup_target,
            warmup_steps=warmup_steps,
            alpha=alpha,
        )

    def _cyclical_lr(self)-> tf.keras.optimizers.schedules.LearningRateSchedule:
        return tfa.optimizers.CyclicalLearningRate(
            initial_learning_rate=self.initial_lr,
            maximal_learning_rate=self.max_lr,
            step_size=2 * self.steps_per_epoch,
            scale_fn=lambda x: 1 / (2.0 ** (x - 1)),
        )

    def _cosine_decay_restarts(self)->tf.keras.optimizers.schedules.LearningRateSchedule:
        first_decay_steps = self.steps_per_epoch * 4
        alpha = self.min_lr / self.max_lr
        return tf.keras.experimental.CosineDecayRestarts(
            initial_learning_rate=self.initial_lr,
            first_decay_steps=first_decay_steps,
            t_mul=self.t_mul,
            m_mul=self.m_mul,
            alpha=alpha,
        )

    def get(self):
        """Returns the configured learning rate scheduler."""
        return self.scheduler