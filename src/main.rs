use anyhow::Result;
use candle_core::{Device, Tensor, DType};
use candle_nn::{VarMap, VarBuilder, Optimizer, Module};
use rand::Rng;
use serde::Serialize;
use std::fs::File;
use std::io::Write;

// --- 1. DATA GENERATION ---

fn to_binary_vec(val: u64, num_bits: usize) -> Vec<f32> {
    let mut vec = vec![0.0; num_bits];
    for i in 0..num_bits {
        vec[i] = ((val >> i) & 1) as f32;
    }
    vec
}

/// Generates Collatz transition pairs: (n_t, n_{t+1})
fn generate_dataset(num_samples: usize, num_steps: usize, num_bits: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    let mut rng = rand::thread_rng();
    let mut x_t_data = Vec::with_capacity(num_samples * num_steps * num_bits);
    let mut x_next_data = Vec::with_capacity(num_samples * num_steps * num_bits);

    for _ in 0..num_samples {
        // Start from a random number in [2, 2^32] to avoid overflow and trivial 0/1 states
        let mut val = rng.gen_range(2..4_294_967_296u64);
        for _ in 0..num_steps {
            let next_val = if val % 2 == 0 {
                val / 2
            } else {
                val.checked_mul(3).and_then(|v| v.checked_add(1)).unwrap_or(1)
            };

            let x_t_bin = to_binary_vec(val, num_bits);
            let x_next_bin = to_binary_vec(next_val, num_bits);

            x_t_data.extend_from_slice(&x_t_bin);
            x_next_data.extend_from_slice(&x_next_bin);

            val = next_val;
        }
    }

    let total_samples = num_samples * num_steps;
    let x_t = Tensor::from_vec(x_t_data, (total_samples, num_bits), device)?;
    let x_next = Tensor::from_vec(x_next_data, (total_samples, num_bits), device)?;
    Ok((x_t, x_next))
}

// --- 2. NETWORK DEFINITIONS ---

struct Encoder {
    linear1: candle_nn::Linear,
    linear2: candle_nn::Linear,
    linear3: candle_nn::Linear,
}

impl Encoder {
    fn new(vs: VarBuilder) -> Result<Self> {
        let linear1 = candle_nn::linear(64, 128, vs.pp("linear1"))?;
        let linear2 = candle_nn::linear(128, 128, vs.pp("linear2"))?;
        let linear3 = candle_nn::linear(128, 2, vs.pp("linear3"))?;
        Ok(Self { linear1, linear2, linear3 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear1.forward(x)?;
        let x = x.relu()?;
        let x = self.linear2.forward(&x)?;
        let x = x.relu()?;
        let x = self.linear3.forward(&x)?;
        
        // CÚ HACK: Chuẩn hóa L2 về đường tròn đơn vị (Unit Circle) để tránh suy biến về [0,0]
        let norm = x.sqr()?.sum_keepdim(1)?.sqrt()?;
        let x = x.broadcast_div(&(norm + 1e-8)?)?;
        Ok(x)
    }
}

struct Decoder {
    linear1: candle_nn::Linear,
    linear2: candle_nn::Linear,
    linear3: candle_nn::Linear,
}

impl Decoder {
    fn new(vs: VarBuilder) -> Result<Self> {
        let linear1 = candle_nn::linear(2, 128, vs.pp("linear1"))?;
        let linear2 = candle_nn::linear(128, 128, vs.pp("linear2"))?;
        let linear3 = candle_nn::linear(128, 64, vs.pp("linear3"))?;
        Ok(Self { linear1, linear2, linear3 })
    }

    fn forward(&self, z: &Tensor) -> Result<Tensor> {
        let z = self.linear1.forward(z)?;
        let z = z.relu()?;
        let z = self.linear2.forward(&z)?;
        let z = z.relu()?;
        let z = self.linear3.forward(&z)?;
        Ok(z)
    }
}

struct CollatzHackerModel {
    encoder: Encoder,
    decoder: Decoder,
    phi: Tensor,
}

impl CollatzHackerModel {
    fn new(vs: VarBuilder) -> Result<Self> {
        let encoder = Encoder::new(vs.pp("encoder"))?;
        let decoder = Decoder::new(vs.pp("decoder"))?;
        
        // Phi là tham số tự do dùng để tính toán Theta qua hàm Sigmoid
        let phi = vs.get_with_hints(
            (1,),
            "phi",
            candle_nn::Init::Const(0.0), // Bắt đầu ở giữa [min, max]
        )?;
        Ok(Self { encoder, decoder, phi })
    }

    fn get_theta(&self) -> Result<Tensor> {
        // RÀNG BUỘC PHẠM VI GÓC (Patch 1): ép theta trong khoảng [15°, 345°]
        let min_theta = 15.0f32.to_radians();
        let max_theta = 345.0f32.to_radians();
        
        let phi_scalar = self.phi.reshape(())?;
        // sigmoid(x) = 1 / (1 + exp(-x))
        let sig_phi = (((phi_scalar * -1.0)?.exp()? + 1.0)?.recip())?;
        
        let min_t = Tensor::new(min_theta, self.phi.device())?;
        let range_t = Tensor::new(max_theta - min_theta, self.phi.device())?;
        let scaled_phi = sig_phi.broadcast_mul(&range_t)?;
        let theta = min_t.broadcast_add(&scaled_phi)?;
        Ok(theta)
    }

    /// Áp dụng ma trận xoay 2D từ góc theta hiện tại
    fn rotate(&self, z: &Tensor) -> Result<Tensor> {
        // z: [batch_size, 2]
        let x = z.narrow(1, 0, 1)?;
        let y = z.narrow(1, 1, 1)?;

        let theta = self.get_theta()?;
        let cos_t = theta.cos()?;
        let sin_t = theta.sin()?;

        // z_pred_x = x*cos + y*sin
        // z_pred_y = -x*sin + y*cos
        let pred_x = (x.broadcast_mul(&cos_t)? + y.broadcast_mul(&sin_t)?)?;
        let pred_y = ((x.broadcast_mul(&sin_t)? * -1.0)? + y.broadcast_mul(&cos_t)?)?;

        let z_pred = Tensor::cat(&[pred_x, pred_y], 1)?;
        Ok(z_pred)
    }

    fn forward(&self, x_t: &Tensor, x_next: &Tensor) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let z_t = self.encoder.forward(x_t)?;
        let z_next = self.encoder.forward(x_next)?;

        let x_t_recon = self.decoder.forward(&z_t)?;
        let x_next_recon = self.decoder.forward(&z_next)?;

        let z_t_pred = self.rotate(&z_t)?;

        Ok((x_t_recon, x_next_recon, z_t_pred, z_next, z_t))
    }

    /// Dịch ngược trọng số của Encoder (Patch 4: Weights Reverse Engineering)
    /// Tính toán ma trận trọng số hiệu dụng W_eff = W3 * W2 * W1 (kích thước [2, 64])
    /// để tìm ra các bit đầu vào nào có ảnh hưởng mạnh nhất đến tọa độ latent (X, Y)
    fn print_weights_analysis(&self) -> Result<()> {
        let w1 = self.encoder.linear1.weight();
        let w2 = self.encoder.linear2.weight();
        let w3 = self.encoder.linear3.weight();

        // W_eff = W3 * W2 * W1
        // [2, 128] * [128, 128] -> [2, 128]
        // [2, 128] * [128, 64] -> [2, 64]
        let w3_w2 = w3.matmul(w2)?;
        let w_eff = w3_w2.matmul(w1)?;

        let w_eff_vec = w_eff.to_vec2::<f32>()?;
        let w_x = &w_eff_vec[0]; // Ảnh hưởng lên tọa độ X (trục ngang)
        let w_y = &w_eff_vec[1]; // Ảnh hưởng lên tọa độ Y (trục dọc)

        println!("\n=== WEIGHTS REVERSE ENGINEERING (LINEAR APPROXIMATION) ===");
        println!("Phân tích mức độ ảnh hưởng của từng bit nhị phân đầu vào (2^0 -> 2^63) lên không gian ẩn:");

        // Sắp xếp mức độ ảnh hưởng theo trị tuyệt đối giảm dần cho X
        let mut x_inf: Vec<(usize, f32)> = w_x.iter().enumerate().map(|(i, &w)| (i, w)).collect();
        x_inf.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

        // Sắp xếp mức độ ảnh hưởng theo trị tuyệt đối giảm dần cho Y
        let mut y_inf: Vec<(usize, f32)> = w_y.iter().enumerate().map(|(i, &w)| (i, w)).collect();
        y_inf.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

        println!("\nTop 10 bits ảnh hưởng mạnh nhất tới trục X (Trục Đông Nam - Tây Bắc):");
        for i in 0..10 {
            let (bit, weight) = x_inf[i];
            let math_symbol = if bit == 0 { "LSB (Lẻ/Chẵn)".to_string() } else { format!("2^{}", bit) };
            println!("  Bit {:2} ({:<13}): Trọng số = {:+.6}", bit, math_symbol, weight);
        }

        println!("\nTop 10 bits ảnh hưởng mạnh nhất tới trục Y (Trục dao động con lắc):");
        for i in 0..10 {
            let (bit, weight) = y_inf[i];
            let math_symbol = if bit == 0 { "LSB (Lẻ/Chẵn)".to_string() } else { format!("2^{}", bit) };
            println!("  Bit {:2} ({:<13}): Trọng số = {:+.6}", bit, math_symbol, weight);
        }

        Ok(())
    }
}

// --- 3. LOSS FUNCTIONS ---

/// Hàm tính Binary Cross Entropy với Logits ổn định số học
fn binary_cross_entropy_with_logits(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let max_val = logits.relu()?;
    let logits_targets = (logits * targets)?;
    let abs_logits = logits.abs()?;
    let neg_abs_logits = (abs_logits * -1.0)?;
    let exp_neg_abs = neg_abs_logits.exp()?;
    let log_1_exp = (exp_neg_abs + 1.0)?.log()?;
    
    let loss = (max_val - logits_targets + log_1_exp)?;
    let mean_loss = loss.mean_all()?;
    Ok(mean_loss)
}

fn mse_loss(pred: &Tensor, target: &Tensor) -> Result<Tensor> {
    let diff = (pred - target)?;
    let mse = diff.sqr()?.mean_all()?;
    Ok(mse)
}

/// Hàm phạt phân tán (Patch 2: Repulsion/Contrastive Loss) để tránh AI tụ thành một cụm rác
fn repulsion_loss(z: &Tensor) -> Result<Tensor> {
    let batch_size = z.dim(0)?;
    if batch_size <= 1 {
        return Ok(Tensor::new(0.0f32, z.device())?);
    }
    // gram = z * z^T (shape [B, B])
    let gram = z.matmul(&z.transpose(0, 1)?)?;
    let sum_squares = gram.sqr()?.sum_all()?;
    
    // Frobenius norm squared minus sum of diagonal (which is batch_size)
    let batch_size_t = Tensor::new(batch_size as f32, z.device())?;
    let off_diag_sum = (sum_squares - &batch_size_t)?;
    
    // Average over all off-diagonal elements: B * (B - 1)
    let num_elements = (batch_size * (batch_size - 1)) as f32;
    let num_elements_t = Tensor::new(num_elements, z.device())?;
    let loss = (&off_diag_sum / &num_elements_t)?;
    Ok(loss)
}

fn compute_loss(
    model: &CollatzHackerModel,
    x_t: &Tensor,
    x_next: &Tensor,
    alpha: f32,
    beta: f32,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let (x_t_recon, x_next_recon, z_t_pred, z_next, z_t) = model.forward(x_t, x_next)?;

    // Reconstruction Loss
    let recon_loss_t = binary_cross_entropy_with_logits(&x_t_recon, x_t)?;
    let recon_loss_next = binary_cross_entropy_with_logits(&x_next_recon, x_next)?;
    let recon_loss = ((recon_loss_t + recon_loss_next)? * 0.5)?;

    // Dynamic Loss
    let dynamic_loss = mse_loss(&z_t_pred, &z_next)?;

    // Repulsion Loss (Patch 2)
    let rep_loss = repulsion_loss(&z_t)?;

    // Total Loss = Recon + alpha * Dyn + beta * Rep
    let alpha_tensor = Tensor::new(alpha, x_t.device())?;
    let beta_tensor = Tensor::new(beta, x_t.device())?;
    let total_loss = (recon_loss.clone() 
        + (dynamic_loss.clone() * &alpha_tensor)? 
        + (rep_loss.clone() * &beta_tensor)?)?;

    Ok((total_loss, recon_loss, dynamic_loss, rep_loss))
}

// --- 4. VISUALIZATION AND REPORTING ---

#[derive(Serialize)]
struct TrajectoryPoint {
    step: usize,
    val: u64,
    x: f32,
    y: f32,
}

#[derive(Serialize)]
struct Trajectory {
    start_val: u64,
    points: Vec<TrajectoryPoint>,
}

#[derive(Serialize)]
struct TrainingReport {
    loss_history: Vec<f32>,
    recon_loss_history: Vec<f32>,
    dyn_loss_history: Vec<f32>,
    rep_loss_history: Vec<f32>,
    theta_history: Vec<f32>,
    trajectories: Vec<Trajectory>,
    final_theta: f32,
}

fn generate_html_report(report: &TrainingReport, filepath: &str) -> Result<()> {
    let template = include_str!("report_template.html");
    let serialized_data = serde_json::to_string(report)?;
    
    let html_content = template.replace("REPORT_DATA_PLACEHOLDER", &serialized_data);
    
    let mut file = File::create(filepath)?;
    file.write_all(html_content.as_bytes())?;
    Ok(())
}

// --- 5. MAIN TRAINING LOOP ---

fn main() -> Result<()> {
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("=== Collatz Latent Space Autoencoder Probe (Rust Version) ===");
    println!("Device: {:?}", device);

    // Hyperparameters
    let num_samples = 5000; // Dataset size (Patch 3)
    let num_steps = 20;
    let num_bits = 64;
    let batch_size = 512;
    let epochs = 150; // Increased epochs for SGD convergence
    let lr = 0.02; // Learning rate for SGD (Patch 3)
    let alpha = 0.5f32; // Tỉ trọng của loss động lực học
    let beta = 0.5f32;  // Tỉ trọng của loss phân tán (Patch 2)

    println!("Generating Collatz sequences dataset...");
    let (x_t, x_next) = generate_dataset(num_samples, num_steps, num_bits, &device)?;
    let dataset_size = x_t.dim(0)?;
    println!("Dataset generated. Total transitions: {}", dataset_size);

    // Initialize parameters
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = CollatzHackerModel::new(vs)?;

    // Config Optimizer (Patch 3: SGD thuần chủng)
    let mut opt = candle_nn::SGD::new(varmap.all_vars(), lr)?;

    // History log
    let mut loss_history = Vec::new();
    let mut recon_loss_history = Vec::new();
    let mut dyn_loss_history = Vec::new();
    let mut rep_loss_history = Vec::new();
    let mut theta_history = Vec::new();

    println!("Starting training for {} epochs...", epochs);
    for epoch in 1..=epochs {
        let mut epoch_loss = 0.0;
        let mut epoch_recon = 0.0;
        let mut epoch_dyn = 0.0;
        let mut epoch_rep = 0.0;
        let mut num_batches = 0;

        // Shuffle indices
        let mut indices: Vec<usize> = (0..dataset_size).collect();
        let mut rng = rand::thread_rng();
        use rand::seq::SliceRandom;
        indices.shuffle(&mut rng);

        for chunk in indices.chunks(batch_size) {
            let chunk_len = chunk.len();
            let chunk_u32: Vec<u32> = chunk.iter().map(|&idx| idx as u32).collect();
            let batch_indices = Tensor::from_slice(&chunk_u32, chunk_len, &device)?;
            let batch_x_t = x_t.index_select(&batch_indices, 0)?;
            let batch_x_next = x_next.index_select(&batch_indices, 0)?;

            let (loss, recon, dyn_l, rep_l) = compute_loss(&model, &batch_x_t, &batch_x_next, alpha, beta)?;

            opt.backward_step(&loss)?;

            epoch_loss += loss.to_vec0::<f32>()?;
            epoch_recon += recon.to_vec0::<f32>()?;
            epoch_dyn += dyn_l.to_vec0::<f32>()?;
            epoch_rep += rep_l.to_vec0::<f32>()?;
            num_batches += 1;
        }

        let avg_loss = epoch_loss / num_batches as f32;
        let avg_recon = epoch_recon / num_batches as f32;
        let avg_dyn = epoch_dyn / num_batches as f32;
        let avg_rep = epoch_rep / num_batches as f32;
        let theta_val = model.get_theta()?.to_vec0::<f32>()?;
        let theta_deg = theta_val.to_degrees();

        loss_history.push(avg_loss);
        recon_loss_history.push(avg_recon);
        dyn_loss_history.push(avg_dyn);
        rep_loss_history.push(avg_rep);
        theta_history.push(theta_deg);

        if epoch % 10 == 0 || epoch == 1 {
            println!(
                "Epoch {:3}/{}: Loss = {:.6} (Recon = {:.6}, Dyn = {:.6}, Rep = {:.6}) | Theta = {:.2}° ({:.4} rad)",
                epoch, epochs, avg_loss, avg_recon, avg_dyn, avg_rep, theta_deg, theta_val
            );
        }
    }

    let final_theta = model.get_theta()?.to_vec0::<f32>()?.to_degrees();
    println!("Training completed! Final Theta: {:.2}°", final_theta);

    // Generate trajectories for visualization
    println!("Generating test trajectories...");
    let test_starts = vec![27u64, 12u64, 1000u64, 7u64, 8521u64];
    let mut trajectories = Vec::new();

    for &start in &test_starts {
        let mut points = Vec::new();
        let mut val = start;
        for step in 0..=20 {
            let bin_vec = to_binary_vec(val, num_bits);
            let bin_tensor = Tensor::from_slice(&bin_vec, (1, num_bits), &device)?;
            let z = model.encoder.forward(&bin_tensor)?;
            let z_vec = z.to_vec2::<f32>()?[0].clone();

            points.push(TrajectoryPoint {
                step,
                val,
                x: z_vec[0],
                y: z_vec[1],
            });

            val = if val % 2 == 0 {
                val / 2
            } else {
                val.checked_mul(3).and_then(|v| v.checked_add(1)).unwrap_or(1)
            };
        }
        trajectories.push(Trajectory {
            start_val: start,
            points,
        });
    }

    let report = TrainingReport {
        loss_history,
        recon_loss_history,
        dyn_loss_history,
        rep_loss_history,
        theta_history,
        trajectories,
        final_theta,
    };

    let report_path = "/home/huy/Documents/my_test/report.html";
    println!("Saving interactive HTML report to {}", report_path);
    generate_html_report(&report, report_path)?;
    println!("Dashboard successfully created!");

    // Chạy phân tích dịch ngược trọng số
    model.print_weights_analysis()?;

    Ok(())
}
