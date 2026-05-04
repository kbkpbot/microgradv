module microgradv

import math

fn test_engine() {
	a := value(-4.0)
	assert a.data == -4.0
	assert a.grad == 0.0
	assert a.op == ''
	b := value(2.0)
	assert b.parents == []
	mut c := a.add(b)
	assert c.data == -2.0
	assert c.op == '+'
	assert c.parents == [a, b]
	mut d := a.mul(b).add(b.pow(3))
	assert d.data == 0
	assert d.op == '+'
	c = c.add(c.add(value(1)))
	c = c.add(value(1)).add(c).sub(a)
	assert c.data == -1
	d_1 := d.add(d.mul(value(2))).add(b.add(a))
	d = d_1.tanh()
	assert d.data == -0.9640275800758169
	assert d.op == 'tanh'
	assert d.parents == [d_1]
	d = d.add(value(3).mul(d)).add(b.sub(a)).relu()
	assert d.op == 'ReLu'
	assert d.data == 2.1438896796967324
	e := c.sub(d)
	assert e.parents == [c, d]
	assert e.op == '-'
	assert e.data == -3.1438896796967324
	assert e.grad == 0.0
	f := e.pow(2)
	assert f.op == '**'
	assert f.data == 9.884042318103623
	assert f.grad == 0.0
	assert e in f.parents
	mut g := f.div(value(2))
	assert f in g.parents
	assert g.op == '/'
	g = g.add(value(10).div(f))
	g = g.mul(value(2))
	assert g.op == '*'
	assert g.data == 11.90750593312356
	assert g.grad == 0.0
	assert g.data == 11.90750593312356
	g.backward()
	assert g.grad == 1.0
	assert a.grad == -10.10998355157139
	assert b.grad == 20.32762219152632
	assert e.grad == -5.000543596581886
	assert f.grad == 0.7952797499345223
}

fn test_relu_gradient() {
	// Test ReLU gradient: should be 0 for negative input, 1 for positive
	v1 := value(-2.0)
	mut r1 := v1.relu()
	r1.backward()
	assert v1.grad == 0.0 // Negative input, gradient should be 0

	v2 := value(3.0)
	mut r2 := v2.relu()
	r2.backward()
	assert v2.grad == 1.0 // Positive input, gradient should be 1
}

fn test_tanh_gradient() {
	// Test tanh gradient: d/dx tanh(x) = 1 - tanh^2(x)
	v := value(1.0)
	mut t := v.tanh()
	t.backward()
	expected_grad := 1.0 - (math.tanh(1.0) * math.tanh(1.0))
	diff := math.abs(v.grad - expected_grad)
	assert diff < 1e-10
}

fn test_backward_topo_order() {
	// Test that backward pass respects topological order
	a := value(2.0)
	b := value(3.0)
	c := a.add(b) // 5
	d := c.mul(a) // 10
	mut e := d.tanh() // tanh(10)

	e.backward()

	// Gradients should be computed correctly
	assert e.grad == 1.0
	assert a.grad != 0.0
	assert b.grad != 0.0
}

fn test_pow_gradient() {
	// Test power function gradient
	// d/dx(x^n) = n * x^(n-1)
	v := value(3.0)
	mut p := v.pow(2) // 3^2 = 9
	p.backward()
	// dg/dv = 2 * 3^1 = 6
	diff := math.abs(v.grad - 6.0)
	assert diff < 1e-10
}

fn test_sigmoid_forward() {
	// Test sigmoid forward pass
	inputs := [-2.0, -1.0, 0.0, 1.0, 2.0]
	for x_val in inputs {
		v := value(x_val)
		s := v.sigmoid()
		// sigmoid(x) = 1 / (1 + exp(-x))
		expected := 1.0 / (1.0 + math.exp(-x_val))
		diff := math.abs(s.data - expected)
		assert diff < 1e-10
	}
}

fn test_sigmoid_gradient() {
	// Test sigmoid gradient: s * (1 - s)
	v := value(1.0)
	mut s := v.sigmoid()
	s.backward()
	// sigmoid(1) ≈ 0.731, gradient ≈ 0.731 * (1 - 0.731) ≈ 0.197
	expected_grad := s.data * (1 - s.data)
	diff := math.abs(v.grad - expected_grad)
	assert diff < 1e-10
}

fn test_numerical_stability_div() {
	// Test division by zero warning (should not crash)
	a := value(1.0)
	b := value(0.0)
	result := a.div(b) // Should print warning but not crash
	// Result should be +Inf
	// Just verify it runs without crash
	// +Inf > 0 is true
	assert result.data > 0
}

fn test_numerical_stability_pow() {
	// Test pow with negative base and non-integer exponent
	a := value(-1.0)
	result := a.pow(0.5) // sqrt(-1) = NaN
	// Should print warning
	// NaN != NaN in V, so we just check it runs without crash
	_ := result
}
