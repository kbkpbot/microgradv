// Basic math operations using method calls
import microgradv

fn main() {
	println('=== Basic Math Operations ===')

	// Create values
	a := microgradv.value(3.0)
	b := microgradv.value(4.0)
	c := microgradv.value(2.0)

	// Using method calls
	println('Using methods:')
	sum := a.add(b)
	println('  ${a.data} + ${b.data} = ${sum.data}')

	diff := a.sub(b)
	println('  ${a.data} - ${b.data} = ${diff.data}')

	prod := a.mul(b)
	println('  ${a.data} * ${b.data} = ${prod.data}')

	quot := a.div(c)
	println('  ${a.data} / ${c.data} = ${quot.data}')

	// Complex expression with methods
	mut expr := (a.add(b)).mul(c).sub((a.div(c)).pow(2))
	println('\nComplex expression: (a + b) * c - (a / c)^2')
	println('  Result: ${expr.data}')

	// Backward pass
	expr.backward()
	println('\nGradients after backward():')
	println('  da = ${a.grad:.4}')
	println('  db = ${b.grad:.4}')
	println('  dc = ${c.grad:.4}')
}
