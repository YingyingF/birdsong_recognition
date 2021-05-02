# Birdsong Recognition
The idea for this projec is to build a small webapp so that one can record bird songs/calls on the phone and have a trained model predict what bird it is. I'm hoping to use this to get quick feedback as I learn more about bird calls in my neighbourhood.


I'm also trying out nbdev to see how useful it can be.


What I like about nbdev:

- The idea of focusing on the notebook and keeping code, documentation and tests in the same place. 


After experimenting with nbdev for a little while, I decided not to continue this project with this nbdev template for a few reasons:

- #export cell is a nice feature, but it's too easy for notebook local variables to sneak in to the notebook cell accidentally, so the exported function ends up with errors when used independently.
- The idea of including tests in the notebook is nice in that it's convenient, but because it's in a notebook, it doesn't force you to write standalone tests.

Maybe I'll give it another try in the future.