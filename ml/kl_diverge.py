import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- simple teacher (bigger) and student (smaller) architectures ---
class TeacherNet(nn.Module):
    def __init__(self, input_dim=32, hidden=256, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes)
        )
    def forward(self, x): return self.net(x)

class StudentNet(nn.Module):
    def __init__(self, input_dim=32, hidden=64, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes)
        )
    def forward(self, x): return self.net(x)

# --- distillation loss function ---
def distillation_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.5):
    """
    student_logits: (B, C)
    teacher_logits: (B, C)  (teacher must be in eval mode)
    labels: LongTensor (B,)
    T: temperature
    alpha: weight for hard-label CE (in [0,1])
    """
    # Hard target loss (student vs true labels)
    ce_loss = F.cross_entropy(student_logits, labels)

    # Soft target loss (teacher vs student using KL)
    # student log-probs at temperature T
    student_log_probs = F.log_softmax(student_logits / T, dim=1)
    # teacher probs (as targets) at temperature T (no log)
    teacher_probs = F.softmax(teacher_logits / T, dim=1)

    kld_loss = nn.KLDivLoss(reduction="batchmean")(student_log_probs, teacher_probs)

    # Combine (note the T^2 factor)
    loss = alpha * ce_loss + (1.0 - alpha) * (T * T) * kld_loss
    return loss, ce_loss.item(), kld_loss.item()

# --- tiny training loop on synthetic data ---
def train_demo(epochs=10, batch_size=64):
    # synthetic dataset: inputs of dim 32, 10 classes
    X = torch.randn(2000, 32)
    y = torch.randint(0, 10, (2000,))
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)

    teacher = TeacherNet().to(device)
    student = StudentNet().to(device)

    # Pretend teacher is pre-trained: we train teacher briefly here for demo
    optim_t = torch.optim.Adam(teacher.parameters(), lr=1e-3)
    teacher.train()
    for _ in range(20):  # quick pretrain (demo only)
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optim_t.zero_grad()
            logits_t = teacher(bx)
            loss_t = F.cross_entropy(logits_t, by)
            loss_t.backward()
            optim_t.step()
    teacher.eval()  # freeze teacher

    optim_s = torch.optim.Adam(student.parameters(), lr=1e-3)

    T = 4.0
    alpha = 0.7  # weight on hard labels; (1-alpha) on distillation term

    for epoch in range(1, epochs + 1):
        student.train()
        total_loss = 0.0
        total_ce = 0.0
        total_kld = 0.0
        n = 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            with torch.no_grad():
                logits_t = teacher(bx)  # teacher in eval, no grad

            logits_s = student(bx)

            loss, ce_val, kld_val = distillation_loss(logits_s, logits_t, by, T=T, alpha=alpha)

            optim_s.zero_grad()
            loss.backward()
            optim_s.step()

            batch_size_actual = bx.size(0)
            total_loss += loss.item() * batch_size_actual
            total_ce += ce_val * batch_size_actual
            total_kld += kld_val * batch_size_actual
            n += batch_size_actual

        print(f"Epoch {epoch}: Loss={total_loss/n:.4f}, CE={total_ce/n:.4f}, KLD={total_kld/n:.4f}")

    return teacher, student

if __name__ == "__main__":
    train_demo(epochs=6)
