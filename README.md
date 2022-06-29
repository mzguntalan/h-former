# H-Former

# Overview
`H-Former` is neural network designed to generate in-between fonts by modeling the latent space of fonts/glyphs which 
allows it to combine several fonts into one called an `in between font` of the given fonts. It is based on a Variational
Autoencoder (VAE) architecture to learn the latent space of fonts conditioned on different glyph identities. It uses an
encoder which works on a set of patches of points that embeds each patch into a vector using a simplified point net, then
uses a transformer on the patch vectors together with vectors for the glyph identity and a couple of learned vectors called
here as holder variables which lets the transformer store intermediate results - the first of the holder variable is
used as a raw code vector which is then used to derive the mean and log variance corresponding to the region of the latent
space that an input glyph resides in - the reparameterization trick is used to sample this region for a code vector. The
decoder is composed of several independent AtlasV2 networks which each learns an elementary structure it learns to 
deform using the code vector from the encoder - the union of the points these make reconstruct a glyph. H-Former was 
trained for 300 epochs using an Adam optimizer with gradient clipping by norm with a loss that is the sum of: 1) chamfer
distance between the original and reconstructed glyphs; 2) KL divergence loss on the means and log variances of the 
Encoder; 3) and an adjustment loss the punishes the decoder for have big weights - these losses are weighted as 1000, 
1, and 1, respectively.

## Glyph
A glyph a character, usually purposeful to humans. In this context we refer to the upper case and lower case letters 
$\[A-Z\]\[a-z\]$ as the 52 glyphs ids or glyph identities (a certain glyph can be identified as an ‘A’, ‘b’, etc.

### Glyphs as a Point Sequence
A glyph is closed figure, possibly with wholes, but for the purposes of this project we are going to view them a point 
sequence. Specifically, a glyph is the point sequence generated by the 
[vegetable project](https://github.com/mzguntalan/vegetable).  

## Font
A font is a style manifested in a set of glyphs. In this context we refer to the set of 52 stylized glyphs 
as one `font`.

# Inspiration and Motivation
This project makes use of point sequences because of my fondness with point data. As stated above, a glyph is a set of 
points, but not just any arrangement of points - it has to be just the right arrangement for you to be able to identify 
it as its appropriate identity (this is analogous to legible vs illegible handwriting). H-former was designed to a 
generator that produces mostly legible fonts.

## Font Tweening and VAE
There are two popular ways of generation: VAE and GAN architectures. This project makes use of the VAE architecture 
to enable `H-former` to “add” fonts together. Roughly speaking, if I had two fonts $f_1$ and $f_2$, a VAE would 
allow us to to $50\%f_1 + 50\%f_2$ - this could also be done with an arbitrary number of fonts. This means font 
generation happens by producing in-between fonts which I would call here as `font tweening`.

## Results
Here is a gif showing it transition through many 32 fonts with 20 in betweens per pair.

![animated-font-tweening-by-h-former](docs/images/h-former-font-generation-demo.gif)

# Setup
A conda environment config file has been provided. Run the following command to create the environment called `h-former`.

```bash
conda env create -f environment.yaml
```

Use the following to activate this environment.

```bash
conda activate h-former
```

Use the following to add this project to your python path variable (temporarily)
```bash
export PYTHONPATH=${PWD}:${PYTHONPATH}
```

## Remarks
- This project uses the trinity `jax`-`flax`-`optax` as its deep learning and acceleration backbone. 
- This project uses `tensorflow` for its data loading and batching. 
- `matplotlib` and `imageio` are for generating the visualizations

## Training
You can train from scratch on the very limited sample dataset by:
- Going to `config.py`
- Change `Config.dataset_filename` to `"./data/fonts-demo.tfrecords"`
- Change `Config.model_weights_filename` to something different to not overwrite the current weights provided
- On the terminal run 
```shell
python ./training/train.py
```

## Generating animations
You can generate animations like the one shown in Section Results, by using the script on creating animations 
(`./visuzalization/create_animation.py`). This 
script loads a random set of fonts fromt the dataset and tweens between them pair (this also loops the last font to 
the first font to create a seamless transition). However, if you want to create specific animations, refer to that 
script and the module ./visualization/animation to create those.

### `create_animation` script
To generate an animation using the script, you will need to:
- Go to `config.py`
- Tweak the `Config.animation_x` parameters to your liking
- Do note to specify the number of fonts not greater than what is in the actual dataset
- Run 
```shell
python ./visualization/create_animation.py
```

## Other (implicit) requirements / caveats
Since this project involves libraries with very picky requirements such as `jax` and `flax` on CUDA and cuDNN on your 
system, problems can/will occur if you:
- don't have a GPU compatible with `jax` since the code makes use of `jax`'s just in time compilation called `jit`
- linux systems are of best compatibility for the dependencies of this projects
- mac systems might not be able to run the `jax` related aspects at all
- windows systems might be able to run this, but installing dependencies might not be as simple as just making the 
conda environment by the above command; i think there are ways to build `jax` from source, but I am not sure for other
dependencies like `flax` and `optax`; you may certainly use WSL for this but do note that this subsystem hosting takes
at least 50% of your RAM by default.

The CUDA and cuDNN versions I use are the following:
- CUDA: 11.x
- cuDNN: 8.x


# Dataset
The dataset comes from Google Fonts ([read about google fonts here](https://fonts.google.com/about) | 
[you can download them here](https://github.com/google/fonts)). It is processed using  
[vegetable project](https://github.com/mzguntalan/vegetable) and each glyph is rendered as `128` points. 

Here is an example visual from that project.

![sample-data-made-from-vegetable](https://github.com/mzguntalan/vegetable/blob/main/demo/glyph_as_point_sequence.png?raw=true)


This means that each font is a tensor of the shape $\[52, 128, 2\]$.

A sample dataset containing a few random google fonts has been provided under `./data/fonts-demo.tfrecords` which 
contains only 16 fonts.

# Notation
- I use $A: shape$ to denote shapes of matrices into of $A\in R^{shape}$ which superscripts the shape and may make it 
harder to see - all tensors are real-valued anyway.

# Conception of Design
In this section, I discuss architectures, concepts, designs, and researches,  which have informed the designs of several 
aspects of this project. I also talk about the architecture in general of `H-Former`.

## Patches and Point Net
Recently, the [Vision Transformer](https://arxiv.org/pdf/2010.11929.pdf) has shown state of the art performance on 
many datasets and it has found a way to 
break through the computational consumption of transformers on pixels by grouping pixels together into patches. ViT 
then turns each patch of grid of pixels into just one vector which then a transformer is used for the resulting set 
of patch embeddings. 

Using the same spirit, Given a sequence of points $P: \[s, 2\]$, I separate them into patches of $p$ points as 
$P’:\[p,\frac{s}{p},2\]$. To each patch $P’_i: \[s\div p, 2\]$, I use a 
[simplified point net](https://arxiv.org/pdf/1612.00593.pdf) called `SimplifiedPointNet` 
(I give a name to this, so that I may detail its architecture in a latter section called Implementation Details) 
to turn the patches into vectors $V_i: \[e\]$. 

With the above patching and `SimplifiedPointNet` we not have a set of patch vectors $V: \[p,e\]$ for each sequence of 
points. 

## Encoding and the Transformer
Given a glyph $G$, its identity $g$, we first turn the $G$ into a set of patch vectors $V_G$ and (apply an Embedding 
and MLP called `GlyphEmbed_Encoder`) to $g$ to get a vector called $v_g$. 

We also learn $h$ number of `holder variables` these “extra memory spaces” that the transformer can use for its 
interemediate calculations - also the first of these holder variables will be used as our VAE code later on.

These holder variables are vectors of the same size as $V_G$ and $v_g$ which means that each holder variable is of 
such shape $H_i: \[e\]$ 

Together, we have:

- $V_G: \[p, e\]$
- $v_g: \[e\]$ which we expand to $v_g: \[1, e\]$
- $H: \[h, e\]$

We concatenate $V_G, v_g$ and $H$ at their first dimension and get set of context vectors $C: \[h+1+p\]$. 

We then use a transformer called `TransformerEncoder` on $C$ and get back contextualized vectors $C’$. We then take 
what had become of the first holder variable $H’_1$ and apply an MLP called `MLP_holder` after which we apply an MLP 
`MLP_mu` and `MLP_logvar` to $H’_1$ and obtain $\mu: \[e\]$ and $\ln\sigma^2: \[e\]$.

## Reparametrization
As with [VAEs](https://arxiv.org/pdf/1906.02691.pdf), we use the reparametrization trick on $\mu$ and $\ln\sigma^2$ 
whenever we are in training, and just use $\mu$, otherwise.

## Decoding with Atlas V2
The AtlasV2 is a term I have seen used in the paper on [Canonical Capsules](https://arxiv.org/pdf/2012.04718.pdf), 
albeit in the [paper on learned elementary structures](https://arxiv.org/pdf/1908.04725.pdf), I have not read such term 
- instead, it is an architecture that was inspired of [Atlas Net](https://arxiv.org/pdf/1802.05384.pdf) which works 
- like paper mache, but instead of predefine shapes to mache with, this new architecture leans these patches in 
- training.

I do something similar with what they have done in [Canonical Capsules](https://arxiv.org/pdf/2012.04718.pdf) 
wherein they used several Atlas V2 as decoders. 

Given the code vector for the glyph $\mu’:\[e\]$ and the identity $g:\[1\]$, I first apply a `GlyphEmbed_decoder` to 
$g$ to obtain a vector $v_g: \[e\]$. I then concatenate the two to get $c: \[e+e\]$. With $c$ I use one Atlas V2 net 
to decode $p$ points ($p$ refers to the number of patches). 

Since there are $s$ total number of points in a glyph, I use $\frac{s}{p}$ independent Atlas V2 nets whose aggregate 
is its glyph reconstruction for the code $\mu’$ and identity $g$. 

# Discussion
## Decoders Learned to Be A Patch
Similar to what happened to Canonical Capusles, the decoders have each learned to become a specific part of each glyph
which we can refer to as a patch. I use 32 patches/decoders for this and this is the 32 colors you see on the animation.
This means that on the glyph of "A" and "B" (and others) the same colored points came from the same decoder, albeit
with different glyph identities supplied. 

## Analogous parts
Since we can see the patches in colors, we can infer what `H-Former` thinks as analogous parts for different glyphs.

### Very similar letters for its uppercase and lowercase glyphs
For example, refer to the uppercase 'W' and lowercase 'w' glyphs, most fonts would have these very similarly styled, 
with a variation in scale (scale may not be evident in the dataset, since I deliberately scaled each glyph so that for
a box whose 2 opposite corners are $(-1,-1)$ and $(1,1)$ either a glyph fills it along the horizontal direction without
overextending along the vertical direction or fills it along the vertical direction without overextending along the
other direction). Notice that the colors indicate that which parts `H-Former` thinks are analogous which is very obvious
in this example of 'w'. Other exmaples of this would be on 'C'-'c', 'P'-'p', among others.

### Glyphs that are "degenerate" versions of another
Another interesting point of view, is taking a look at different letters, say the glyph for uppercase 'P' and 
uppercase 'R' and notice that intuitively I (and hopefully, other people as well) that, for most fonts, the R is a P, 
but with some extra appendage and so the rest of them would be analogous - `H-Former` thinks of this as well, as evident
in the similarly colors in their patchings. This is why I think of this pair as one being a "degenerate" case of the
other - the P is an R with a very very short other leg.

You may notice this phenomenon on other pairs as well: some of them would be
'E'-'F', 'C'-'G', 'J'-'I. 

### Completely different glyphs
An even more bizzare observation to make, would be comparing glyphs that are wildly different, say a 'Q' and a 'K'. 
Since there are no intuitively analog parts for these, it is interesting to know what `H-Former` thinks their analog
parts are. 

You can have fun at imagining how two glyph parts are analogous - while it might be silly for some, I think
there is something we can learn here that `H-Former` has learned and shows us through the patching its made. Take for 
example the two-storey lowercase 'a' and the lowercase 'b'. They are completely visually different, but take a look at 
the patchings `H-Former` has made, it has colored the hole in the 'a' and 'b' the same, then notice (also imagine) you 
can take the ascender (upper portion of the left stem of 'b') and morph it to the right and curving it to create the 
upper curve of 'a'.

### Analogous part in different Fonts
Up until now, I've only been sharing about anologous parts of glyphs in the same font, but we can also observe what 
`H-Former` thinks are analogous parts for different fonts. 

## Transitions and Tweening
We generally see smooth transitions for fonts that are similar and rough transitions when the fonts are too different
such as in the case of cursive to non-cursive. 

## What does 2 (or more) fonts look like when combined
With `H-Former`, we can make up in between fonts, which means we can combine different fonts together and this can be
a key source of inspiration for typographers and the like.

`H-Former` seems to do very well with non-cursive fonts, probably because it is easier to map analogous parts between 
them. It might be a case that there is not enough parameters on `H-former` for it to capture very intricate and unique 
details that may not transfer over to most fonts since cursive fonts tends to have little in common with the rest. 

# Implementation Details
## SimplifiedPointNet
This is a round of several MLP/Norm layers as follows:

- features=768, activation=relu
- features=512, activation=relu
- features=256, activation=relu
- LayerNorm
- features=256, activation=relu

## GlyphEmbed_Encoder and GlyphEmbed_Decoder
This is a round of MLP layers as follows:

- features=256, activation=relu
- features=256, activation=relu

## MLP_holder
This is 2 layers of MLP as follows:

- features=256, activation=relu
- features=256, activation=relu

## MLP_mu, MLP_logvar
This is a single layer of a linear layer with

- features=256

## TransformerEncoder
I use the following config:

- number of heads: 8
- embed dim: 256
- feedforward dim: 512 (in the paper of [Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf) 
they use a 2 layer MLP inside the transformer block, here the 2 layers are both ReLU-activated and are 512 and 256 in 
features, respectively)
- number of transformer blocks: 3

## Atlas V2
### Learned Elementary Structure
It first learns to a set of 4 vectors whose shape is $\[8\]$. It then uses an MLP to convert each of them into 2-d 
points whose layers are as follows:

- features=256, activation=relu
- features=256, activation=relu
- features=2, activation=tanh

## Learned Adjustment
This takes a context vector $c: \[256\]$ and a patch of points $\[4,2\]$ . I then concatenate them to derive a tensor of 
shape $\[4,2+256\]$ then use an MLP with the following layers:

- features=256, activation=relu
- features=256, activation=relu
- features=256, activation=relu
- features=2, activation=tanh

## Training
I train using the adam optimizer with the same learning schedule as used in 
[Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) which is also shown here in the 
[tensorflow docs](https://www.tensorflow.org/text/tutorials/transformer) - additionally, I clip the gradients so that 
their norm is at max 1.0. 

I train on 300 epochs with the following losses:

- `main`: a main loss that is computed as the [chamfer distance](http://3ddl.cs.princeton.edu/2016/slides/su.pdf) between the 
- `kl_div`: original glyphs and the reconstructions [KL divergence loss](https://arxiv.org/pdf/1906.02691.pdf)
- `adj`: adjustment loss that punishes the Atlas V2’s layers for having big weights - this encourages the adjustments to be 
simpler; this is the sum of the absolute values and of the square values of these weights

I combine the above losses into one loss by the following weights $L = 1000main + kl_{div} + adj$

### Batches
Every batch consists of 8 fonts, which means $8\times52$ glyphs in total.

# Author Details and Citing this project
```shell
@software{Untalan_H_Former_2020,
  author = {Untalan, Marko Zolo G.},
  month = {6},
  title = {{H-Former}},
  url = {TODO},
  year = {2022}
}
```