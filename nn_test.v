module microgradv

import math

fn test_neuron_creation() {
	// Test neuron creation
	n := new_neuron(3)
	assert n.weights.len == 3
	assert n.bias.data >= -1.0 && n.bias.data <= 1.0
	for w in n.weights {
		assert w.data >= -1.0 && w.data <= 1.0
	}
}

fn test_neuron_forward() {
	n := new_neuron(2)
	input := [value(1.0), value(2.0)]
	output := n.forward(input) or {
		assert false
		return
	}
	// Output should be tanh of (w1*x1 + w2*x2 + b)
	sum := n.weights[0].data * 1.0 + n.weights[1].data * 2.0 + n.bias.data
	expected := math.tanh(sum)
	assert output.data == expected
}

fn test_neuron_forward_wrong_dim() {
	n := new_neuron(3)
	input := [value(1.0), value(2.0)] // Wrong dimension
	result := n.forward(input) or { return }
	_ := result
	// Should have returned early due to error
	assert false
}

fn test_neuron_parameters() {
	n := new_neuron(2)
	params := n.parameters()
	// Should have 2 weights + 1 bias = 3 parameters
	assert params.len == 3
	assert n.bias in params
	assert n.weights[0] in params
	assert n.weights[1] in params
}

fn test_layer_creation() {
	l := new_layer(3, 5) // 3 inputs, 5 neurons
	assert l.neurons.len == 5
	assert l.dimensions == [3, 5]
}

fn test_layer_forward() {
	l := new_layer(2, 3)
	input := [value(1.0), value(2.0)]
	output := l.forward(input) or {
		assert false
		return
	}
	assert output.len == 3 // 3 neurons, 3 outputs
	for o in output {
		// Each output should be a Value (just check it's not nil)
		assert o != 0
	}
}

fn test_layer_parameters() {
	l := new_layer(2, 3)
	params := l.parameters()
	// 3 neurons * (2 weights + 1 bias) = 9 parameters
	assert params.len == 9
}

fn test_mlp_creation() {
	mlp := new_mlp(3, [4, 2, 1]) // 3 inputs, two hidden layers (4, 2), 1 output
	assert mlp.layers.len == 3
	assert mlp.dimensions == [3, 4, 2, 1]
}

fn test_mlp_forward() {
	mlp := new_mlp(2, [3, 1]) // 2 inputs, 3 neurons hidden, 1 output
	input := [value(1.0), value(2.0)]
	output := mlp.forward(input) or {
		assert false
		return
	}
	assert output.len == 1 // Final layer has 1 neuron
}

fn test_mlp_parameters() {
	// MLP: 2 inputs -> 3 neurons -> 1 neuron
	// Layer 1: 3 * (2+1) = 9 params
	// Layer 2: 1 * (3+1) = 4 params
	// Total: 13 params
	mlp := new_mlp(2, [3, 1])
	params := mlp.parameters()
	assert params.len == 13
}

fn test_training_simple() {
	// Simple test: learn to compute y = tanh(x1 + x2)
	// Using tanh activation, so outputs are bounded to [-1, 1]
	mlp := new_mlp(2, [4, 1])

	// Training data: y = tanh(0.5*(x1 + x2)) scaled to [-1, 1]
	inputs := [
		[value(-2.0), value(-2.0)],
		[value(-1.0), value(0.0)],
		[value(0.0), value(1.0)],
		[value(1.0), value(1.0)],
	]
	// Targets in [-1, 1] range
	targets := [value(-0.99), value(-0.46), value(0.46), value(0.76)]

	lr := 0.5

	// Train for more epochs
	for epoch in 0 .. 200 {
		mut params := mlp.parameters()
		for mut p in params {
			p.grad = 0.0
		}

		// Forward and loss
		mut loss := value(0.0)
		for i in 0 .. inputs.len {
			pred := mlp.forward(inputs[i]) or { continue }
			diff := pred[0].sub(targets[i])
			loss = loss.add(diff.pow(2))
		}

		loss.backward()

		for mut p in params {
			p.data -= lr * p.grad
		}
	}

	// Test the trained network
	test_input := [value(2.0), value(0.0)]
	pred := mlp.forward(test_input) or {
		assert false
		return
	}
	// For input (2, 0), tanh(0.5*(2+0)) ≈ tanh(1) ≈ 0.76
	expected := 0.76
	diff := math.abs(pred[0].data - expected)
	assert diff < 0.3
}
