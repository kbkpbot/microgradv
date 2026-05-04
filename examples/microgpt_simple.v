// microGPT simplified example using microgradv
// This demonstrates character-level language modeling with a simple MLP
module main

import microgradv
import os
import rand

fn main() {
	println('=== microGPT Simplified Example ===')

	// Load a few names for training
	content := os.read_file('input.txt') or { panic('input.txt not found') }
	lines := content.split('\n')
	mut docs := []string{}
	for line in lines {
		trimmed := line.trim_space()
		if trimmed.len > 0 && docs.len < 50 {
			docs << trimmed
		}
	}
	println('Loaded ${docs.len} names for training')

	// Simple char set
	mut chars := map[string]bool{}
	for doc in docs {
		for ch in doc {
			chars['${ch}'] = true
		}
	}
	mut vocab := []string{}
	for ch, _ in chars {
		vocab << ch
	}
	vocab.sort()
	vocab_size := vocab.len
	println('Vocab size: ${vocab_size}')

	// Create simple model: char -> embedding -> MLP -> next char logits
	embedding_dim := 16
	hidden_dim := 32

	// Parameters: embedding matrix and MLP weights
	mut embed := [][]&microgradv.Value{}
	for _ in 0 .. vocab_size {
		mut row := []&microgradv.Value{}
		for _ in 0 .. embedding_dim {
			row << microgradv.value(rand.f64_in_range(-0.1, 0.1) or { 0.01 })
		}
		embed << row
	}

	mut w1 := [][]&microgradv.Value{}
	for _ in 0 .. hidden_dim {
		mut row := []&microgradv.Value{}
		for _ in 0 .. embedding_dim {
			row << microgradv.value(rand.f64_in_range(-0.1, 0.1) or { 0.01 })
		}
		w1 << row
	}

	mut b1 := []&microgradv.Value{}
	for _ in 0 .. hidden_dim {
		b1 << microgradv.value(0.0)
	}

	mut w2 := [][]&microgradv.Value{}
	for _ in 0 .. vocab_size {
		mut row := []&microgradv.Value{}
		for _ in 0 .. hidden_dim {
			row << microgradv.value(rand.f64_in_range(-0.1, 0.1) or { 0.01 })
		}
		w2 << row
	}

	// Get all parameters
	mut params := []&microgradv.Value{}
	for row in embed {
		for p in row {
			params << p
		}
	}
	for row in w1 {
		for p in row {
			params << p
		}
	}
	for p in b1 {
		params << p
	}
	for row in w2 {
		for p in row {
			params << p
		}
	}
	println('Model parameters: ${params.len}')

	// Training
	lr := 0.5
	epochs := 3

	println('\nTraining for ${epochs} epochs...')
	for epoch in 0 .. epochs {
		mut total_loss_data := 0.0
		mut n_examples := 0

		for doc in docs {
			// Zero gradients
			for mut p in params {
				p.grad = 0.0
			}

			// Convert doc to char ids
			mut tokens := []int{}
			for ch in doc {
				for i, c in vocab {
					if c == '${ch}' {
						tokens << i
						break
					}
				}
			}

			// Train on each position
			mut doc_loss := microgradv.value(0.0)
			for i in 0 .. tokens.len - 1 {
				// Get embedding of current char
				mut x := embed[tokens[i]]

				// MLP: x -> hidden (with ReLU)
				mut h := []&microgradv.Value{}
				for j in 0 .. hidden_dim {
					mut sum := b1[j]
					for k in 0 .. x.len {
						sum = sum.add(x[k].mul(w1[j][k]))
					}
					h << sum.relu()
				}

				// Output: h -> vocab logits
				mut logits := []&microgradv.Value{}
				for j in 0 .. vocab_size {
					mut sum := microgradv.value(0.0)
					for k in 0 .. h.len {
						sum = sum.add(h[k].mul(w2[j][k]))
					}
					logits << sum
				}

				// Softmax
				mut max_val := logits[0].data
				for l in logits {
					if l.data > max_val {
						max_val = l.data
					}
				}
				mut exps := []&microgradv.Value{}
				for l in logits {
					exps << l.sub(microgradv.value(max_val)).exp()
				}
				mut total_exp := microgradv.value(0.0)
				for e in exps {
					total_exp = total_exp.add(e)
				}
				mut probs := []&microgradv.Value{}
				for e in exps {
					probs << e.div(total_exp)
				}

				// Loss: -log(probs[target])
				target_id := tokens[i + 1]
				loss_t := probs[target_id].log().mul(microgradv.value(-1.0))
				doc_loss = doc_loss.add(loss_t)
				n_examples++
			}

			// Backward
			doc_loss.backward()

			// Update parameters
			for mut p in params {
				p.data -= lr * p.grad
			}

			total_loss_data += doc_loss.data / f64(tokens.len - 1)
		}

		avg_loss := total_loss_data / f64(docs.len)
		println('Epoch ${epoch}: avg loss = ${avg_loss:.4}')
	}

	println('\nDone! Model trained.')

	// Generate samples
	println('\n--- Generated Names ---')
	for sample_idx in 0 .. 5 {
		mut name_chars := []string{}
		// Start with random character
		mut token_id := rand.int_in_range(0, vocab_size) or { 0 }

		for _ in 0 .. 15 {
			// Get embedding
			x := embed[token_id]

			// MLP forward
			mut h := []&microgradv.Value{}
			for j in 0 .. hidden_dim {
				mut sum := b1[j]
				for k in 0 .. x.len {
					sum = sum.add(x[k].mul(w1[j][k]))
				}
				h << sum.relu()
			}

			mut logits := []&microgradv.Value{}
			for j in 0 .. vocab_size {
				mut sum := microgradv.value(0.0)
				for k in 0 .. h.len {
					sum = sum.add(h[k].mul(w2[j][k]))
				}
				logits << sum
			}

			// Softmax
			mut max_val := logits[0].data
			for l in logits {
				if l.data > max_val {
					max_val = l.data
				}
			}
			mut exps := []&microgradv.Value{}
			for l in logits {
				exps << l.sub(microgradv.value(max_val)).exp()
			}
			mut total := microgradv.value(0.0)
			for e in exps {
				total = total.add(e)
			}
			mut probs := []f64{}
			for e in exps {
				probs << e.data / total.data
			}

			// Sample
			r := rand.f64()
			mut cumsum := 0.0
			mut next_id := 0
			for i, p in probs {
				cumsum += p
				if cumsum >= r {
					next_id = i
					break
				}
			}

			if next_id >= vocab.len {
				break
			}
			// Convert single char string to running string
			name_chars << vocab[next_id]
			token_id = next_id
		}

		println('  ${sample_idx + 1:2d}: ${name_chars.join('')}')
	}
}
