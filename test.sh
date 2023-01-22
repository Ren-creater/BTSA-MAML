#----------  Test MAML Cifar -----------------------
# CUDA_VISIBLE_DEVICES="0" python train_maml.py --train=False --dataset cifar100 --meta_batch_size 4 --update_batch_size 1 --update_lr 0.01 --num_updates 5 --num_classes 5 --logdir logs/maml_cifar5w1s --num_test_tasks 600 --num_filters 32 --max_pool True 
# CUDA_VISIBLE_DEVICES="1" python train_maml.py --train=False --dataset cifar100 --meta_batch_size 4 --update_batch_size 5 --update_lr 0.01 --num_updates 5 --num_classes 5 --logdir logs/maml_cifar5w5s --num_test_tasks 600 --num_filters 32 --max_pool True --num_groups 5 
# CUDA_VISIBLE_DEVICES="2" python train_maml.py --train=False --dataset cifar100 --meta_batch_size 4 --update_batch_size 1 --update_lr 0.01 --num_updates 5 --num_classes 10 --logdir logs/maml_cifar10w1s --num_test_tasks 600 --num_filters 32 --max_pool True --num_groups 5
# CUDA_VISIBLE_DEVICES="3" python train_maml.py --train=False --dataset cifar100 --meta_batch_size 4 --update_batch_size 5 --update_lr 0.01 --num_updates 5 --num_classes 10 --logdir logs/maml_cifar10w5s --num_test_tasks 600 --num_filters 32 --max_pool True --num_groups 5

#----------  Test BTSA-MAML Cifar -----------------------
# CUDA_VISIBLE_DEVICES="5" python train_btsamaml.py --train=False --dataset cifar100  --meta_batch_size 4 --update_batch_size 1 --update_lr 0.01 --num_updates 5 --num_classes 5 --logdir logs/tsa_maml_cifar5w1s --premaml logs/maml_cifar5w1s/bestmodel --num_test_tasks 600 --num_filters 32 --max_pool True --num_groups 5 --cosann True
# CUDA_VISIBLE_DEVICES="6" python train_btsamaml.py --train=False --dataset cifar100  --meta_batch_size 4 --update_batch_size 5 --update_lr 0.01 --num_updates 5 --num_classes 5 --logdir logs/tsa_maml_cifar5w5s --premaml logs/maml_cifar5w5s/bestmodel --num_test_tasks 600 --num_filters 32 --max_pool True --num_groups 10 --cosann True
# CUDA_VISIBLE_DEVICES="7" python train_btsamaml.py --train=False --dataset cifar100  --meta_batch_size 4 --update_batch_size 1 --update_lr 0.01 --num_updates 5 --num_classes 10 --logdir logs/tsa_maml_cifar10w1s --premaml logs/maml_cifar10w1s/bestmodel --num_test_tasks 600 --num_filters 32 --max_pool True --num_groups 5 --cosann True
# CUDA_VISIBLE_DEVICES="8" python train_btsamaml.py --train=False --dataset cifar100  --meta_batch_size 4 --update_batch_size 5 --update_lr 0.01 --num_updates 5 --num_classes 10 --logdir logs/tsa_maml_cifar10w5s --premaml logs/maml_cifar10w5s/bestmodel --num_test_tasks 600 --num_filters 32 --max_pool True --num_groups 5 --cosann True

#----------  Test MAML TieredImage -----------------------
# CUDA_VISIBLE_DEVICES="1" python train_maml.py --train=False --dataset tiered --meta_batch_size 4 --update_batch_size 1 --update_lr 0.01 --num_updates 5 --num_classes 5 --logdir logs/maml_tiered5w1s --num_test_tasks 600 --num_filters 32 --max_pool True 
# CUDA_VISIBLE_DEVICES="2" python train_maml.py --train=False --dataset tiered --meta_batch_size 4 --update_batch_size 5 --update_lr 0.01 --num_updates 5 --num_classes 5 --logdir logs/maml_tiered5w5s --num_test_tasks 600 --num_filters 32 --max_pool True --num_groups 5 
# CUDA_VISIBLE_DEVICES="3" python train_maml.py --train=False --dataset tiered --meta_batch_size 4 --update_batch_size 1 --update_lr 0.01 --num_updates 5 --num_classes 10 --logdir logs/maml_tiered10w1s --num_test_tasks 600 --num_filters 32 --max_pool True --num_groups 5
# CUDA_VISIBLE_DEVICES="4" python train_maml.py --train=False --dataset tiered --meta_batch_size 4 --update_batch_size 5 --update_lr 0.01 --num_updates 5 --num_classes 10 --logdir logs/maml_tiered10w5s --num_test_tasks 600 --num_filters 32 --max_pool True --num_groups 5

#----------  Test BTSA-MAML TieredImage -----------------------
# CUDA_VISIBLE_DEVICES="5" python train_btsamaml.py --train=False --dataset tiered --meta_batch_size 4 --update_batch_size 1 --update_lr 0.01 --num_updates 5 --num_classes 5 --logdir logs/tsa_maml_tiered5w1s --premaml logs/maml_tiered5w1s/bestmodel --num_test_tasks 600 --num_filters 32 --max_pool True --num_groups 5 --cosann True
# CUDA_VISIBLE_DEVICES="6" python train_btsamaml.py --train=False --dataset tiered --meta_batch_size 4 --update_batch_size 5 --update_lr 0.01 --num_updates 5 --num_classes 5 --logdir logs/tsa_maml_tiered5w5s --premaml logs/maml_tiered5w5s/bestmodel --num_test_tasks 600 --num_filters 32 --max_pool True --num_groups 5 --cosann True
# CUDA_VISIBLE_DEVICES="7" python train_btsamaml.py --train=False --dataset tiered --meta_batch_size 4 --update_batch_size 1 --update_lr 0.01 --num_updates 5 --num_classes 10 --logdir logs/tsa_maml_tiered10w1s --premaml logs/maml_tiered10w1s/bestmodel --num_test_tasks 600 --num_filters 32 --max_pool True --num_groups 5 --cosann True
# CUDA_VISIBLE_DEVICES="8" python train_btsamaml.py --train=False --dataset tiered --meta_batch_size 4 --update_batch_size 5 --update_lr 0.01 --num_updates 5 --num_classes 10 --logdir logs/tsa_maml_tiered10w5s --premaml logs/maml_tiered10w5s/bestmodel --num_test_tasks 600 --num_filters 32 --max_pool True --num_groups 5 --cosann True