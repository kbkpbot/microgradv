module microgradv

// TrainConfig holds training configuration
struct TrainConfig {
	lr     f64 = 0.01
	epochs int = 100
}

// train a model using SGD
// inputs: batch of training examples
// targets: expected outputs
// model: MLP to train
pub fn train(model &MLP, inputs [][]&Value, targets []&Value, config TrainConfig) {
	mut params := model.parameters()

	for epoch in 0 .. config.epochs {
		// Zero gradients
		for mut p in params {
			p.grad = 0.0
		}

		// Forward pass and loss
		mut loss := value(0.0)
		for i in 0 .. inputs.len {
			pred := model.forward(inputs[i]) or { continue }
			for j in 0 .. pred.len {
				diff := pred[j].sub(targets[i])
				loss = loss.add(diff.pow(2))
			}
		}

		// Backward pass
		mut loss_mut := loss
		loss_mut.backward()

		// Update parameters
		for mut p in params {
			p.data -= config.lr * p.grad
		}

		if epoch % 50 == 0 {
			println('Epoch ${epoch}: loss = ${loss.data:.6}')
		}
	}
}

// mse_loss computes mean squared error
pub fn mse_loss(predictions []&Value, targets []&Value) &Value {
	mut loss := value(0.0)
	for i in 0 .. predictions.len {
		diff := predictions[i].sub(targets[i])
		loss = loss.add(diff.pow(2))
	}
	return loss.div(value(f64(predictions.len)))
}
