"""
CRC-Select training script with alternating optimization.

Implements the CRC-Select algorithm:
1. Warmup: Train vanilla SelectiveNet
2. Alternating: Calibrate q on D_cal, then train with CRC risk penalty
"""
import os
import sys
from argparse import ArgumentParser

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

from collections import OrderedDict
import torch

from logger_utils.metric import MetricDict
from logger_utils.io import print_metric_dict, save_checkpoint
from logger_utils.logger import Logger

from selectivenet.vgg_variant import vgg16_variant
from selectivenet.model import SelectiveNet
from selectivenet.loss_crc import CRCSelectiveLoss
from selectivenet.data import DatasetBuilder
from selectivenet.data_splits import get_split_loaders
from selectivenet.evaluator import Evaluator
from selectivenet.reproducibility import set_seed

from crc.calibrate import compute_crc_threshold

import wandb
WANDB_PROJECT_NAME = "crc_selective_net"

if "--unobserve" in sys.argv:
    os.environ["WANDB_MODE"] = "dryrun"


def main(args):
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    wandb.init(
        project=WANDB_PROJECT_NAME, 
        tags=["crc-select", "pytorch"],
        config=vars(args)
    )
    train_crc_select(args)


def train_crc_select(args):
    wandb.config.update(args)
    log_path = wandb.run.dir
    
    # Create checkpoint directory for evaluation
    checkpoint_dir = "checkpoints/CRC-Select"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print("=" * 80)
    print("CRC-Select Training")
    print("=" * 80)
    print(f"Seed: {args.seed}")
    print(f"Target coverage: {args.coverage}")
    print(f"Target risk (alpha): {args.alpha_risk}")
    print(f"Warmup epochs: {args.warmup_epochs}")
    print(f"Recalibrate every: {args.recalibrate_every} epochs")
    print("=" * 80)
    
    # ==================== Data ====================
    dataset_builder = DatasetBuilder(name=args.dataset, root_path=args.dataroot)
    full_train_dataset = dataset_builder(
        train=True, normalize=args.normalize, augmentation=args.augmentation
    )
    
    # Get 3-way splits: train/cal/test
    train_loader, cal_loader, test_loader = get_split_loaders(
        full_train_dataset,
        dataset_name=args.dataset,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"\nData splits:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Cal batches: {len(cal_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # ==================== Model ====================
    features = vgg16_variant(dataset_builder.input_size, args.dropout_prob).cuda()
    model = SelectiveNet(
        features, args.dim_features, dataset_builder.num_classes, 
        div_by_ten=args.div_by_ten
    ).cuda()
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    # ==================== Optimizer ====================
    params = model.parameters()
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, 
        weight_decay=args.wd, nesterov=args.nesterov
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma
    )
    
    # ==================== Loss ====================
    base_loss = torch.nn.CrossEntropyLoss(reduction='none')
    crc_loss_fn = CRCSelectiveLoss(
        loss_func=base_loss,
        coverage=args.coverage,
        alpha_risk=args.alpha_risk,
        lm=args.lm,
        alpha=args.alpha,
        mu=args.mu_init
    )
    
    # ==================== Logging ====================
    train_logger = Logger(
        path=os.path.join(log_path, f'train_log_seed{args.seed}.csv'), 
        mode='train'
    )
    val_logger = Logger(
        path=os.path.join(log_path, f'val_log_seed{args.seed}.csv'), 
        mode='val'
    )
    
    # ==================== Training Loop ====================
    best_val_loss = float('inf')
    best_model_state = None
    q = args.alpha_risk  # Initialize q
    
    for epoch in range(args.num_epochs):
        current_phase = 'warmup' if epoch < args.warmup_epochs else 'crc_training'
        
        print(f"\nEpoch {epoch+1}/{args.num_epochs} ({current_phase})")
        print("-" * 80)
        
        # ==================== Phase 1: CRC Calibration ====================
        # Calibrate q on calibration set (only after warmup)
        if epoch >= args.warmup_epochs and epoch % args.recalibrate_every == 0:
            print(f"[Calibration] Running CRC calibration on cal set...")
            
            with torch.no_grad():
                calib_result = compute_crc_threshold(
                    model=model,
                    cal_loader=cal_loader,
                    tau=args.tau,
                    alpha=args.alpha_risk,
                    delta=args.delta,
                    device='cuda'
                )
            
            q = calib_result['q']
            R_cal = calib_result['estimated_risk']
            coverage_cal = calib_result['actual_coverage']
            
            print(f"[Calibration] Results:")
            print(f"  q (threshold): {q:.4f}")
            print(f"  Risk on cal: {R_cal:.4f}")
            print(f"  Coverage on cal: {coverage_cal:.4f}")
            print(f"  Accepted: {calib_result['num_accepted']}/{calib_result['num_total']}")
            
            # Update mu using dual ascent
            if args.use_dual_update:
                old_mu = crc_loss_fn.get_mu()
                crc_loss_fn.update_mu(R_cal, args.alpha_risk, args.dual_lr)
                new_mu = crc_loss_fn.get_mu()
                print(f"[Dual Update] mu: {old_mu:.4f} -> {new_mu:.4f}")
            
            # Log calibration results
            wandb.log({
                'q': q,
                'R_cal': R_cal,
                'coverage_cal': coverage_cal,
                'mu': crc_loss_fn.get_mu(),
                'epoch': epoch
            })
        
        # ==================== Phase 2: Training ====================
        train_metric_dict = MetricDict()
        
        # Disable CRC penalty during warmup
        if current_phase == 'warmup':
            crc_loss_fn.set_mu(0.0)
        
        for i, (x, t) in enumerate(train_loader):
            model.train()
            x = x.to('cuda', non_blocking=True)
            t = t.to('cuda', non_blocking=True)
            
            # Forward
            out_class, out_select, out_aux = model(x)
            
            # Loss and metrics
            loss_dict = crc_loss_fn(
                out_class, out_select, out_aux, t,
                alpha_risk=args.alpha_risk,
                threshold=args.tau,
                mode='train'
            )
            
            loss = loss_dict['loss']
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Convert loss to item for logging
            loss_dict['loss'] = loss.detach().cpu().item()
            train_metric_dict.update(loss_dict)
        
        # Log training metrics
        wandb.log({**train_metric_dict.avg, 'epoch': epoch})
        
        # ==================== Validation ====================
        val_metric_dict = MetricDict()
        
        with torch.autograd.no_grad():
            for i, (x, t) in enumerate(test_loader):
                model.eval()
                x = x.to('cuda', non_blocking=True)
                t = t.to('cuda', non_blocking=True)
                
                # Forward
                out_class, out_select, out_aux = model(x)
                
                # Loss and metrics
                loss_dict_val = crc_loss_fn(
                    out_class, out_select, out_aux, t,
                    alpha_risk=args.alpha_risk,
                    threshold=args.tau,
                    mode='validation'
                )
                
                loss_dict_val['val_loss'] = loss_dict_val['val_loss'].detach().cpu().item()
                
                # Additional evaluation
                evaluator = Evaluator(
                    out_class.detach(), t.detach(), out_select.detach(), 
                    selection_threshold=args.tau
                )
                loss_dict_val.update(evaluator())
                
                val_metric_dict.update(loss_dict_val)
        
        # Log validation metrics
        wandb.log({**val_metric_dict.avg, 'epoch': epoch})
        
        # ==================== Post Epoch ====================
        print_metric_dict(epoch, args.num_epochs, val_metric_dict.avg, mode='val')
        
        train_logger.log(train_metric_dict.avg, step=(epoch+1))
        val_logger.log(val_metric_dict.avg, step=(epoch+1))
        
        # Save best model
        if val_metric_dict.avg['val_loss'] < best_val_loss:
            best_val_loss = val_metric_dict.avg['val_loss']
            best_model_state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'q': q,
                'mu': crc_loss_fn.get_mu()
            }
            print(f"✓ New best model (val_loss: {best_val_loss:.4f})")
        
        scheduler.step()
    
    # ==================== Save Checkpoints ====================
    print("\n" + "=" * 80)
    print("Training completed. Saving checkpoints...")
    
    # Prepare final checkpoint
    final_checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_q': q,
        'final_mu': crc_loss_fn.get_mu()
    }
    
    # Use best_model_state if available, otherwise use final
    if best_model_state is None:
        best_model_state = final_checkpoint.copy()
    
    checkpoint_dict = [
        final_checkpoint,      # Final checkpoint (index 0)
        best_model_state,      # Best validation checkpoint (index 1)
        best_model_state,      # Best validation checkpoint duplicate (index 2)
    ]
    
    # Save to wandb directory
    wandb_checkpoint_path = os.path.join(log_path, 'checkpoints')
    save_checkpoint(checkpoint_dict, wandb_checkpoint_path)
    print(f"✓ Checkpoints saved to wandb: {wandb_checkpoint_path}")
    
    # Save to main checkpoint directory for evaluation
    eval_checkpoint_path = f"checkpoints/CRC-Select/seed_{args.seed}.pth"
    torch.save(checkpoint_dict, eval_checkpoint_path)
    print(f"✓ Checkpoint saved for evaluation: {eval_checkpoint_path}")
    print("=" * 80)


if __name__ == '__main__':
    parser = ArgumentParser()
    
    # Model
    parser.add_argument('--dim_features', type=int, default=512)
    parser.add_argument('--dropout_prob', type=float, default=0.3)
    parser.add_argument('--div_by_ten', action='store_true', 
                       help='divide by 10 when calculating g')
    
    # Data
    parser.add_argument('-d', '--dataset', type=str, default='cifar10')
    parser.add_argument('--dataroot', type=str, default='../data', 
                       help='path to dataset root')
    parser.add_argument('-j', '--num_workers', type=int, default=8)
    parser.add_argument('-N', '--batch_size', type=int, default=128)
    parser.add_argument('--normalize', action='store_false')
    parser.add_argument('--augmentation', type=str, default='original',
                       help='type of augmentation: original, tf, or lili')
    parser.add_argument('--seed', type=int, default=42,
                       help='random seed for reproducibility')
    
    # Optimization
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', action='store_true')
    parser.add_argument('--scheduler_step', type=int, default=25,
                       help='step size for learning rate scheduler')
    parser.add_argument('--scheduler_gamma', type=float, default=0.5,
                       help='gamma for learning rate scheduler')
    
    # SelectiveNet loss
    parser.add_argument('--coverage', type=float, default=0.8,
                       help='target coverage')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='mix weight between selective_loss and ce_loss')
    parser.add_argument('--lm', type=float, default=32.0,
                       help='Lagrange multiplier for coverage constraint')
    
    # CRC-Select specific
    parser.add_argument('--alpha_risk', type=float, default=0.1,
                       help='target risk level for CRC')
    parser.add_argument('--tau', type=float, default=0.5,
                       help='acceptance threshold for selector')
    parser.add_argument('--warmup_epochs', type=int, default=20,
                       help='number of warmup epochs without CRC penalty')
    parser.add_argument('--recalibrate_every', type=int, default=5,
                       help='recalibrate q every N epochs')
    parser.add_argument('--mu_init', type=float, default=1.0,
                       help='initial value for risk penalty multiplier')
    parser.add_argument('--dual_lr', type=float, default=0.01,
                       help='learning rate for dual update of mu')
    parser.add_argument('--delta', type=float, default=0.1,
                       help='failure probability for CRC bound')
    parser.add_argument('--use_dual_update', action='store_true',
                       help='use dual ascent to update mu')
    
    # Logging
    parser.add_argument('-s', '--suffix', type=str, default='')
    parser.add_argument('-l', '--log_dir', type=str, default='../logs/train_crc')
    parser.add_argument('--unobserve', action='store_true',
                       help='disable Weights & Biases')
    
    args = parser.parse_args()
    main(args)

