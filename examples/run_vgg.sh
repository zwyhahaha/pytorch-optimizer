#!/usr/bin/env bash

# python examples/vgg_train.py --cuda --device 1 --model logreg --method sgd     --save --epochs 10 --alpha_0 0.001 --beta 0.001
# python examples/vgg_train.py --cuda --device 2 --model logreg --method sgd_hd  --save --epochs 10 --alpha_0 0.001 --beta 0.001 --parallel --silent
# python examples/vgg_train.py --cuda --device 3 --model logreg --method sgdn    --save --epochs 10 --alpha_0 0.001 --beta 0.001 --parallel --silent
# python examples/vgg_train.py --cuda --device 4 --model logreg --method sgdn_hd --save --epochs 10 --alpha_0 0.001 --beta 0.001 --parallel --silent
# python examples/vgg_train.py --cuda --device 7 --model logreg --method adam    --save --epochs 10 --alpha_0 0.001 --beta 1e-7 --parallel --silent
# python examples/vgg_train.py --cuda --device 1 --model logreg --method adam_hd --save --epochs 10 --alpha_0 0.001 --beta 1e-7 --parallel --silent

# python examples/vgg_train.py --cuda --device 2 --model mlp --method sgd     --save --epochs 100 --alpha_0 0.001 --beta 0.001 --parallel --silent
# python examples/vgg_train.py --cuda --device 3 --model mlp --method sgd_hd  --save --epochs 100 --alpha_0 0.001 --beta 0.001 --parallel --silent
# python examples/vgg_train.py --cuda --device 4 --model mlp --method sgdn    --save --epochs 100 --alpha_0 0.001 --beta 0.001 --parallel --silent
# python examples/vgg_train.py --cuda --device 7 --model mlp --method sgdn_hd --save --epochs 100 --alpha_0 0.001 --beta 0.001 --parallel --silent
# python examples/vgg_train.py --cuda --device 1 --model mlp --method adam    --save --epochs 100 --alpha_0 0.001 --beta 1e-7 --parallel --silent
# python examples/vgg_train.py --cuda --device 2 --model mlp --method adam_hd --save --epochs 100 --alpha_0 0.001 --beta 1e-7 --parallel --silent

# python examples/vgg_train.py --cuda --device 3 --model vgg --method sgd     --save --epochs 100 --alpha_0 0.001 --beta 0.001 --parallel --silent
# python examples/vgg_train.py --cuda --device 4 --model vgg --method sgd_hd  --save --epochs 100 --alpha_0 0.001 --beta 0.001 --parallel --silent
# python examples/vgg_train.py --cuda --device 7 --model vgg --method sgdn    --save --epochs 100 --alpha_0 0.001 --beta 0.001 --parallel --silent
# python examples/vgg_train.py --cuda --device 1 --model vgg --method sgdn_hd --save --epochs 100 --alpha_0 0.001 --beta 0.001 --parallel --silent
# python examples/vgg_train.py --cuda --device 2 --model vgg --method adam    --save --epochs 100 --alpha_0 0.001 --beta 1e-8 --parallel --silent
# python examples/vgg_train.py --cuda --device 0 --model vgg --method adam_hd --save --epochs 100 --alpha_0 0.001 --beta 1e-8 --parallel --silent


# python examples/vgg_train.py --cuda --device 2 --model logreg --method osgm --save --epochs 10 --alpha_0 0.01 --beta 1e-8 
# python examples/vgg_train.py --cuda --device 3 --model logreg --method osmm --save --epochs 10 --alpha_0 0.01 --beta 1e-8 --parallel --silent
python examples/vgg_train.py --cuda --device 1 --model mlp --method osgm --save --epochs 100 --alpha_0 0.01 --beta 1e-8 --parallel --silent
python examples/vgg_train.py --cuda --device 4 --model mlp --method osmm --save --epochs 100 --alpha_0 0.01 --beta 1e-8 --parallel --silent
python examples/vgg_train.py --cuda --device 2 --model mlp --method osgm --save --epochs 100 --alpha_0 0.001 --beta 1e-8 --parallel --silent
python examples/vgg_train.py --cuda --device 3 --model mlp --method osmm --save --epochs 100 --alpha_0 0.001 --beta 1e-8 --parallel --silent
# python examples/vgg_train.py --cuda --device 4 --model vgg --method osgm --save --epochs 100 --alpha_0 0.001 --beta 1e-8 --parallel --silent
# python examples/vgg_train.py --cuda --device 7 --model vgg --method osmm --save --epochs 100 --alpha_0 0.001 --beta 1e-8 --parallel --silent