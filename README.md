simplest baseline: bigram language model, loss, generation training the bigram model port our code to a script Building the "self-attention"
version 1: averaging past context with for loops, the weakest form of aggregation the trick in self-attention: matrix multiply as weighted aggregation version 2: using matrix multiply version 3: adding softmax minor code cleanup positional encoding
Implemented Version 4: full self-attention, where every token emits a Query (what am I looking for?) and a Key (what do I contain?), and the Value is what gets communicated.
Understood the key insight: the data-dependent weights replace the uniform averaging from earlier versions, so tokens can now decide which other tokens matter to them.
