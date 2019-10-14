"""
Lookahead optimizer implementation based on https://arxiv.org/abs/1907.08610
"""
from tensorflow.python.eager import context
import tensorflow as tf

class LookaheadOptimizer(tf.train.Optimizer):
    """
    Lookahead optimizer compatible with other tensorflow optimizers.
    This optimizer accepts an optimizer of user's choice to use it as a
    `lookahead` fast optimizer for k steps, and updates other weights by
    linearly interpolating saved weight before k steps to the direction where
    the fast optimizer has reached.
    """

    def __init__(self,
                 fast_optimizer,
                 k=5,
                 alpha=0.5,
                 use_locking=False,
                 name='Lookahead'):
        super().__init__(use_locking, name)
        self._fast_opt = fast_optimizer
        self._k_constant = k
        self._alpha = alpha

        # Tensors
        self._k_t = None
        self._alpha_t = None

    def _get_step(self):
        with tf.init_scope():
            if context.executing_eagerly():
                graph = None
            else:
                graph = tf.get_default_graph()
            return self._get_non_slot_variable('step', graph=graph)

    def _prepare(self):
        # pylint: disable=protected-access
        self._k_t = tf.convert_to_tensor(self._k_constant,
                                         name='k_t',
                                         dtype=tf.int32)
        self._alpha_t = tf.convert_to_tensor(self._alpha, name='alpha_t')
        self._fast_opt._prepare()

    def _create_slots(self, var_list):
        # pylint: disable=protected-access
        """Create slots for each trainable variables in graph.
        Slots make sure that a variable is allocated to closest device
        in which the corresponding variable is located.
        """
        # Make a copy of each variables to store `slow weights`.
        for var in var_list:
            self._get_or_make_slot(var, var, 'slow', self._name)

        first_var = min(var_list, key=lambda x: x.name)
        self._create_non_slot_variable(
            initial_value=0, name='step', colocate_with=first_var)
        self._fast_opt._create_slots(var_list)

    def _apply_dense(self, grad, var):
        # pylint: disable=protected-access
        slow_weight = self.get_slot(var, 'slow')
        step = self._get_step()
        alpha = tf.cast(self._alpha_t, var.dtype.base_dtype)

        # yapf: disable
        update_var = tf.cond(
            tf.equal(tf.floormod(step, self._k_t), 0),
            lambda: tf.group(
                tf.assign(var,
                          tf.assign_add(
                              slow_weight, (var - slow_weight) * alpha))),
            lambda: self._fast_opt._apply_dense(grad, var))
        # yapf: enable

        return tf.group(update_var)

    def _apply_sparse(self, grad, var):
        raise NotImplementedError('Sparse gradients on lookahead optimizer not '
                                  'supported.')

    def _finish(self, update_ops, name_scope):
        with tf.control_dependencies(update_ops):
            step = self._get_step()
            update_step = tf.assign_add(step, 1, use_locking=self._use_locking)

        return tf.group(*update_ops + [update_step], name=name_scope)