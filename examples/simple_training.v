// Simple training example: learn a linear function
import microgradv

fn main() {
	println('=== Simple Training Example ===')
	println('Learning to compute y = tanh(0.5*(x1 + x2))')

	// Use MLP for this task
	mlp := microgradv.new_mlp(2, [4, 1])

	// Training data
	inputs := [
		[microgradv.value(-2.0), microgradv.value(-2.0)],
		[microgradv.value(-1.0), microgradv.value(0.0)],
		[microgradv.value(0.0), microgradv.value(1.0)],
		[microgradv.value(1.0), microgradv.value(1.0)],
	]
	// Targets in [-1, 1] range (tanh outputs)
	targets := [microgradv.value(-0.99), microgradv.value(-0.46),
		microgradv.value(0.46), microgradv.value(0.76)]

	lr := 0.5

	// Training loop
	for epoch in 0 .. 200 {
		mut params := mlp.parameters()

		// Zero gradients
		for mut p in params {
			p.grad = 0.0
		}

		// Forward pass and loss computation
		mut loss := microgradv.value(0.0)
		for i in 0 .. inputs.len {
			pred := mlp.forward(inputs[i]) or { continue }
			diff := pred[0].sub(targets[i])
			loss = loss.add(diff.pow(2))
		}

		// Backward pass
		mut loss_mut := loss
		loss_mut.backward()

		// Update parameters
		for mut p in params {
			p.data -= lr * p.grad
		}

		if epoch % 50 == 0 {
			println('Epoch ${epoch}: loss = ${loss.data:.6}')
		}
	}

	// Test the learned function
	println('\nTesting:')
	test_input := [microgradv.value(2.0), microgradv.value(0.0)]
	pred := mlp.forward(test_input) or { return }
	// For input (2, 0), tanh(0.5*(2+0)) ≈ tanh(1) ≈ 0.76
	println('  f(2, 0) = ${pred[0].data:.4} (expected: ~0.76)')
}
