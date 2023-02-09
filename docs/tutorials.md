---
layout: tutorials
homepage: false
title: Tutorials
---

Here are some notebooks demonstrating basic tasks implemented with libmolgrid.
We demonstrate usage with PyTorch and Keras with the Tensorflow backend, 
and show how to implement and train a few types of models. For examples of
usage with Caffe, we recommend you look at the <a href="https://github.com/gnina/scripts">gnina scripts</a> and <a href="https://github.com/gnina/models">gnina models</a> repositories. 

# Input files

The "types" files expected as input to ExampleProvider are text files where each 
line is a training example.  The first few columns are numerical labels followed
by molecular structure files, like so:
```
1 6.05 0.162643 4kqp/4kqp_rec_0.gninatypes 4kqp/4kqp_min_0.gninatypes
1 6.05 0.216481 4kqp/4kqp_rec_0.gninatypes 4kqp/4kqp_docked_0.gninatypes
0 -6.05 4.28411 4kqp/4kqp_rec_0.gninatypes 4kqp/4kqp_docked_1.gninatypes
0 -6.05 2.50741 4kqp/4kqp_rec_0.gninatypes 4kqp/4kqp_docked_2.gninatypes
0 -6.05 2.78808 4kqp/4kqp_rec_0.gninatypes 4kqp/4kqp_docked_3.gninatypes
```

Structure files are recommended to be relative paths (the `data_root` setting can
be used to specify what they are relative to).  The `gninatype` format is a minimal
file format that contains nothing other than Cartesian coordinates and atom types.
It can only be used if indexed atom types (as opposed to vector types) are used.
These files are generated using the `gninatyper` binary that is part of `gnina`.

Individual `gninatypes` files can be assembled into a single large, efficient to 
access "cache" file using the script [create_caches2.py](https://github.com/gnina/scripts/blob/master/create_caches2.py).
