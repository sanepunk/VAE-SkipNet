from FlaxVAE-SkipNet import Encoder, Decoder, EncoderDecoder
from ImLoad import load_and_preprocess_image
import jax.numpy as jnp

if __name__ == '__main__':
	x = jnp.array(load_and_preprocess_image("cat.jpeg", 256, 256))
	print(jnp.max(x), "max of x")
	print('starting')
	for i in range(500):
		params, opt_state, loss_val = update(params, opt_state, x / 255., x / 255.)
		# print(jnp.max(x / 255.))
		if i % 100 == 0:
			opt, p1, p2 = encoder_decoder.apply(params, x / 255.)
			print(opt.shape)
			plt.imshow(opt[0])
			plt.show()
			print(loss_val)
	print(x.shape)
	plt.imshow(x[0] / 255.)
	plt.show()