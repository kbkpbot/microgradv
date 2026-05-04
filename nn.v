module microgradv

import rand

@[noinit]
struct Neuron {
pub mut:
	// n dimensional weights, or neurons parameters.
	weights []&Value @[required]
	bias    &Value = value(0)
}

// Initialize weights and bias to random values between -1 and 1
pub fn new_neuron(nb_input int) &Neuron {
	return &Neuron{
		weights: []&Value{len: nb_input, cap: nb_input, init: value(rand.f64_in_range(-1, 1) or {
			1 / nb_input
		})}
		bias:    value(rand.f64_in_range(-1, 1) or { 1 / nb_input })
	}
}

// forward pass for the neuron.
// multiply the input to weights and sum them together with the bias.
pub fn (n &Neuron) forward(x []&Value) !&Value {
	// Ensure the input have same dimension as the neurons weights
	if x.len != n.weights.len {
		return error('the input should have the same dimensions as the neuron')
	}
	mut out := n.bias
	for i in 0 .. x.len {
		out = out.add(n.weights[i].mul(x[i]))
	}
	return out.tanh()
}

pub fn (n &Neuron) parameters() []&Value {
	mut out := [n.bias]
	out << n.weights
	return out
}

@[noinit]
struct Layer {
pub:
	// A layer contains n nb of neurons.
	neurons    []&Neuron
	dimensions []int
}

pub fn new_layer(nb_input int, nb_output int) &Layer {
	return &Layer{
		neurons:    []&Neuron{len: nb_output, cap: nb_output, init: new_neuron(nb_input)}
		dimensions: [nb_input, nb_output]
	}
}

// Iterate over the neurons and to forward pass of each with the given input
pub fn (l &Layer) forward(x []&Value) ![]&Value {
	return []&Value{len: l.dimensions[1], cap: l.dimensions[1], init: l.neurons[index].forward(x)!}
}

pub fn (l &Layer) parameters() []&Value {
	mut out := []&Value{cap: (l.dimensions[0] + 1) * l.dimensions[1]}
	for n in l.neurons {
		out << n.parameters()
	}
	return out
}

@[noinit]
struct MLP {
pub:
	dimensions []int
	layers     []&Layer
}

// Initialize the MLP.
// Each layer will take the output of the previous one and pass it's output the next.
pub fn new_mlp(nb_input int, nb_outputs []int) &MLP {
	mut dm := []int{len: 1, cap: nb_outputs.len + 1, init: nb_input}
	dm << nb_outputs
	return &MLP{
		layers:     []&Layer{len: nb_outputs.len, cap: nb_outputs.len, init: new_layer(dm[index], dm[
			index + 1])}
		dimensions: dm
	}
}

// Iterate over the layers and do forward pass of each with the given input.
pub fn (m &MLP) forward(x []&Value) ![]&Value {
	mut out := m.layers[0].forward(x)!
	for i := 1; i < m.layers.len; i++ {
		out = m.layers[i].forward(out)!
	}
	return out
}

pub fn (m &MLP) parameters() []&Value {
	// Calculate the number of parameters to prevent relocation.
	mut nb_p := 0
	for i := 0; i < m.dimensions.len - 1; i++ {
		nb_p += (m.dimensions[i] + 1) * m.dimensions[i + 1]
	}
	mut out := []&Value{cap: nb_p}
	for l in m.layers {
		out << l.parameters()
	}
	return out
}
