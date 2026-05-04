module microgradv

import math

// Value is a wrapper for f64 type.
pub struct Value {
pub:
	// for debugging, the operation that resulted to this Value.
	op string // default empty
pub mut:
	data f64 @[required]
	// The Values that created this value, if any.
	parents []&Value // default empty
	// The gradient with respect to the Value that .backward() was called on.
	grad f64 = 0.0 // default 0.0
}

pub fn (a &Value) add(b &Value) &Value {
	mut out := &Value{
		data:    a.data + b.data
		parents: [a, b]
		op:      '+'
	}
	return out
}

pub fn (a &Value) mul(b &Value) &Value {
	mut out := &Value{
		data:    a.data * b.data
		parents: [a, b]
		op:      '*'
	}
	return out
}

pub fn (a &Value) pow(b f64) &Value {
	mut out := &Value{
		data:    math.pow(a.data, b)
		parents: [a, value(b)]
		op:      '**'
	}
	return out
}

pub fn (a &Value) sub(b &Value) &Value {
	mut out := &Value{
		data:    a.data - b.data
		parents: [a, b]
		op:      '-'
	}
	return out
}

pub fn (a &Value) div(b &Value) &Value {
	mut out := &Value{
		data:    a.data / b.data
		parents: [a, b]
		op:      '/'
	}
	return out
}

pub fn (a &Value) relu() &Value {
	mut rel := a.data
	if a.data < 0 {
		rel = 0
	}
	mut out := &Value{
		data:    rel
		parents: [a]
		op:      'ReLu'
	}
	return out
}

pub fn (a &Value) tanh() &Value {
	mut out := &Value{
		data:    math.tanh(a.data)
		parents: [a]
		op:      'tanh'
	}
	return out
}

fn val_backward(mut child Value) {
	match child.op {
		'+' {
			child.parents[0].grad += child.grad
			child.parents[1].grad += child.grad
		}
		'-' {
			child.parents[0].grad += child.grad
			child.parents[1].grad -= child.grad
		}
		'*' {
			child.parents[0].grad += child.parents[1].data * child.grad
			child.parents[1].grad += child.parents[0].data * child.grad
		}
		'**' {
			child.parents[0].grad += (child.parents[1].data * math.pow(child.parents[0].data,
				child.parents[1].data - f64(1))) * child.grad
		}
		'/' {
			child.parents[0].grad += math.pow(child.parents[1].data, -1) * child.grad
			child.parents[1].grad += f64(-1) * child.parents[0].data * math.pow(child.parents[1].data, -2) * child.grad
		}
		'ReLu' {
			child.parents[0].grad += if child.parents[0].data > 0 { child.grad } else { 0 }
		}
		'tanh' {
			child.parents[0].grad += (1 - child.data * child.data) * child.grad
		}
		else {
			// do nothing
		}
	}
}

pub fn (mut a Value) backward() {
	// build topological order
	// parents will become children.
	mut children := []&Value{}
	mut visited := []&Value{}
	build_topo(a, mut children, mut visited)
	a.grad = 1
	for i := children.len - 1; i >= 0; i-- {
		val_backward(mut children[i])
	}
}

fn build_topo(a &Value, mut children []&Value, mut visited []&Value) {
	if a !in visited {
		visited << a
		for parent in a.parents {
			build_topo(parent, mut children, mut visited)
		}
		children << a
	}
}

pub fn value(a f64) &Value {
	return &Value{
		data: a
	}
}
