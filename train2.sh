#----------  Train MAML Cifar -----------------------
# python train_maml.py --dataset cifar100 --metatrain_iterations 60000 --meta_batch_size 4 --update_batch_size 1 --update_lr 0.01 --num_updates 5 --num_classes 5 --logdir logs/maml_cifar5w1s --num_filters 32 --max_pool=True 
# python train_maml.py --dataset cifar100 --metatrain_iterations 60000 --meta_batch_size 4 --update_batch_size 5 --update_lr 0.01 --num_updates 5 --num_classes 5 --logdir logs/maml_cifar5w5s --num_filters 32 --max_pool=True 
# python train_maml.py --dataset cifar100 --metatrain_iterations 60000 --meta_batch_size 4 --update_batch_size 1 --update_lr 0.01 --num_updates 5 --num_classes 10 --logdir logs/maml_cifar10w1s --num_filters 32 --max_pool=True 
# python train_maml.py --dataset cifar100 --metatrain_iterations 60000 --meta_batch_size 4 --update_batch_size 5 --update_lr 0.01 --num_updates 5 --num_classes 10 --logdir logs/maml_cifar10w5s --num_filters 32 --max_pool=True

#----------  Train MAML TieredImage -----------------------
# python train_maml.py --dataset tiered --metatrain_iterations 150000 --meta_batch_size 4 --update_batch_size 1 --update_lr 0.01 --num_updates 5 --num_classes 5 --logdir logs/maml_tiered5w1s --num_filters 32 --max_pool=True 
# python train_maml.py --dataset tiered --metatrain_iterations 150000 --meta_batch_size 4 --update_batch_size 5 --update_lr 0.01 --num_updates 5 --num_classes 5 --logdir logs/maml_tiered5w5s --num_filters 32 --max_pool=True 
# python train_maml.py --dataset tiered --metatrain_iterations 150000 --meta_batch_size 4 --update_batch_size 1 --update_lr 0.01 --num_updates 5 --num_classes 10 --logdir logs/maml_tiered10w1s --num_filters 32 --max_pool=True 
# python train_maml.py --dataset tiered --metatrain_iterations 150000 --meta_batch_size 4 --update_batch_size 5 --update_lr 0.01 --num_updates 5 --num_classes 10 --logdir logs/maml_tiered10w5s --num_filters 32 --max_pool=True

