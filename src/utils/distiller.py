import tensorflow as tf

class Distiller(tf.keras.Model):

	def __init__(self, model, features, idxs):
		super(Distiller, self).__init__()
		self.model = model
		self.features = tf.convert_to_tensor(features, dtype=tf.float32)
		self.idxs = tf.cast(tf.constant(idxs), dtype=tf.int32)
		self.table = tf.lookup.experimental.DenseHashTable(key_dtype=tf.int32, value_dtype=tf.float32, default_value=tf.zeros(shape=(features.shape[1],), dtype=tf.float32), empty_key=-2, deleted_key=-3, name='table')
		self.table.insert(self.idxs, self.features)

	def compile(self, optimizer, loss, metrics, alpha=10.0, temperature=4.0,):
		super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
		self.student_loss_fn = loss
		self.distillation_loss_fn = tf.keras.losses.MeanAbsoluteError(name='mae_loss', reduction=tf.keras.losses.Reduction.AUTO)
		self.alpha = alpha
		self.temperature = temperature

	def train_step(self, data):
		((x, y), idxs) = data
		supervision_signal = self.table.lookup(tf.cast(idxs, dtype=tf.int32))
		with tf.GradientTape() as tape:
			preds, embeddings = self.model(x, training=True)
			loss = self.student_loss_fn(y, preds)
			distillation_loss = self.distillation_loss_fn(supervision_signal,embeddings)
			r_loss = tf.add_n(self.model.losses) if self.model.losses else tf.constant(0.0)
			loss  = loss + (self.alpha * distillation_loss) + r_loss
		# Compute gradients
		trainable_vars = self.model.trainable_variables
		gradients = tape.gradient(loss, trainable_vars)
		# Update weights
		self.optimizer.apply_gradients(zip(gradients, trainable_vars))
		# Update the metrics configured in `compile()`.
		self.compiled_metrics.update_state(y, preds)
		# Return a dict of performance
		return {"loss": loss, "accuracy": self.metrics[0].result(), "distil_loss": distillation_loss,}

	def test_step(self, data):
		x, y = data
		y_prediction, _ = self.model(x, training=False)
		student_loss = self.student_loss_fn(y, y_prediction)
		self.compiled_metrics.update_state(y, y_prediction)
		results = {m.name: m.result() for m in self.metrics}
		results.update({"loss": student_loss})
		return results



	




def forward_kl(student_logits, teacher_logits, temperature=2.0):
    """FedLN Eq. 9: T² KL(softmax(student/T) || softmax(global/T))."""
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    student_logp = tf.nn.log_softmax(student_logits / temperature, axis=-1)
    teacher_logp = tf.nn.log_softmax(tf.stop_gradient(teacher_logits) / temperature, axis=-1)
    return temperature ** 2 * tf.reduce_mean(tf.reduce_sum(
        tf.exp(student_logp) * (student_logp - teacher_logp), axis=-1))


class GlobalLogitDistiller(tf.keras.Model):
    """Train a classifier against the frozen incoming round-global classifier."""
    def __init__(self, model, teacher, beta=10.0, temperature=2.0):
        super().__init__()
        if beta < 0 or temperature <= 0:
            raise ValueError('beta must be nonnegative and temperature positive')
        self.model, self.teacher = model, teacher
        self.teacher.trainable = False
        self.beta, self.temperature = beta, temperature

    def compile(self, optimizer, loss, metrics):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = loss

    def train_step(self, data):
        x, y = data
        teacher_logits = tf.stop_gradient(self.teacher(x, training=False))
        with tf.GradientTape() as tape:
            predictions = self.model(x, training=True)
            distillation = forward_kl(predictions, teacher_logits, self.temperature)
            regularization = tf.add_n(self.model.losses) if self.model.losses else 0.0
            loss = self.student_loss_fn(y, predictions) + self.beta * distillation + regularization
        variables = self.model.trainable_variables
        self.optimizer.apply_gradients(zip(tape.gradient(loss, variables), variables))
        self.compiled_metrics.update_state(y, predictions)
        return {'loss': loss, 'distil_loss': distillation,
                **{metric.name: metric.result() for metric in self.metrics}}

    def test_step(self, data):
        x, y = data
        predictions = self.model(x, training=False)
        self.compiled_metrics.update_state(y, predictions)
        return {'loss': self.student_loss_fn(y, predictions),
                **{metric.name: metric.result() for metric in self.metrics}}
