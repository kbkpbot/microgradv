# microgradv

A tiny Autograd engine, about 250 lines of code in total!

An implementation of Andrej karpathy's [micrograd](https://github.com/karpathy/micrograd) in Vlang that I wrote as exercise to learn about neural networks and V. 

> [!NOTE]
> Due to V's limitations on operator overloading using this implementation is painful to the eyes.
> I hope this will be useful for educational purposes like the original implementation.

## Installation

```
v install Sydiepus.microgradv
```

## Example Usage

```v
import sydiepus.microgradv

a := microgradv.value(-4.0)
b := microgradv.value(2.0)
mut c := a.add(b)
mut d := a.mul(b).add(b.pow(3))
c = c.add(c.add(microgradv.value(1)))
c = c.add(microgradv.value(1)).add(c).sub(a)
d = d.add(d.mul(microgradv.value(2))).add(b.add(a).relu())
d = d.add(microgradv.value(3).mul(d)).add(b.sub(a).relu())
e := c.sub(d)
f := e.pow(2)
mut g := f.div(microgradv.value(2))
g = g.add(microgradv.value(10).div(f))
println('${g.data:.4}') // prints 24.7041, the outcome of this forward pass
g.backward()
println('${a.grad:.4}') // prints 138.8338, i.e. the numerical value of dg/da
println('${b.grad:.4}') // prints 645.5773, i.e. the numerical value of dg/db
```

Or using method calls:
```v
import sydiepus.microgradv

a := microgradv.value(-4.0)
b := microgradv.value(2.0)
mut c := a.add(b)
mut d := a.mul(b).add(b.pow(3))
c = c.add(c.add(microgradv.value(1)))
c = c.add(microgradv.value(1)).add(c).sub(a)
d = d.add(d.mul(microgradv.value(2))).add(b.add(a).relu())
d = d.add(microgradv.value(3).mul(d)).add(b.sub(a).relu())
e := c.sub(d)
f := e.pow(2)
mut g := f.div(microgradv.value(2))
g = g.add(microgradv.value(10).div(f))
println('${g.data:.4}') // prints 24.7041
g.backward()
println('${a.grad:.4}') // prints 138.8338
println('${b.grad:.4}') // prints 645.5773
```

## Training a neural network

```v
import sydiepus.microgradv

// Create a 3x4x4x1 MLP
mlp := microgradv.new_mlp(3, [4, 4, 1])
// Get the parameters to train
mut mpl_param := mlp.parameters()
// Training data
inputs := [[microgradv.value(2), microgradv.value(3), microgradv.value(-1)],
	[microgradv.value(3), microgradv.value(-1), microgradv.value(0.5)],
	[microgradv.value(0.5), microgradv.value(1), microgradv.value(1)],
	[microgradv.value(1), microgradv.value(1), microgradv.value(-1)]]
// Ground truth
desired_outputs := [microgradv.value(1), microgradv.value(-1),
	microgradv.value(-1), microgradv.value(1)]
// Train
for t := 0; t <= 50; t++ {
	mut forward_passes := []&microgradv.Value{}
	// Predict
	for input in inputs {
		forward_passes << mlp.forward(input)![0]
	}
	// Calculate loss, (prediction - ground truth)^2
	mut loss := (forward_passes[0].sub(desired_outputs[0])).pow(2)
	for i in 1 .. inputs.len {
		loss = loss.add((forward_passes[i].sub(desired_outputs[i])).pow(2))
	}
	// Zero the gradients
	for mut p in mpl_param {
		p.grad = 0.0
	}
	// Do backpropagation
	loss.backward()
	// Print the current iteration, loss, and the predictions
	println('Iter ${t}')
	println('-----------------------------------')
	println('The loss is : ${loss.data}')
	println('-----------------------------------')
	println('The predictions are : ${forward_passes}')
	// The learning rate
	l_rate := 0.05
	// Tune the weights
	for mut p in mpl_param {
		p.data += -l_rate * p.grad
	}
}
```
