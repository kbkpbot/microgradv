// Demonstrate activation functions and their gradients
import microgradv

fn main() {
	println('=== Activation Functions ===')

	// Test different inputs
	inputs := [-2.0, -1.0, 0.0, 1.0, 2.0]

	println('\nReLU:')
	for x in inputs {
		v := microgradv.value(x)
		mut r := v.relu()
		r.backward()
		println('  f(${x}) = ${r.data:.4}, f\'(${x}) = ${v.grad:.4}')
	}

	println('\nTanh:')
	for x in inputs {
		v := microgradv.value(x)
		mut t := v.tanh()
		t.backward()
		println('  f(${x}) = ${t.data:.4}, f\'(${x}) = ${v.grad:.4}')
	}

	// Demonstrate chain rule with activation
	println('\nChain rule example: f(x) = tanh(x * 2 + 1)')
	x := microgradv.value(1.5)
	mut f := (x.mul(microgradv.value(2.0))).add(microgradv.value(1.0)).tanh()
	f.backward()
	println('  f(1.5) = ${f.data:.4}')
	println('  f\'(1.5) = ${x.grad:.4}')
}
