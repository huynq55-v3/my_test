/// Con chip ảo Collatz sử dụng hệ tọa độ ẩn 2D
pub struct CollatzVirtualChip {
    pub val: u64,
    pub latent: [f32; 2],
    pub theta: f32,
}

impl CollatzVirtualChip {
    pub fn new(start_val: u64) -> Self {
        let mut chip = Self {
            val: start_val,
            latent: [0.0, 0.0],
            theta: 0.0,
        };
        chip.latent = chip.encode(start_val);
        chip.theta = chip.dynamic_clock(start_val);
        chip
    }

    /// Khối Encoder hình học: ánh xạ u64 lên vòng tròn đơn vị
    pub fn encode(&self, num: u64) -> [f32; 2] {
        if num == 0 {
            return [0.0, 0.0];
        }

        // 1. Trục X: Phân định Lẻ/Chẵn dựa trên LSB (Bit 0)
        let x_raw = if num % 2 == 1 {
            // Số Lẻ: Bán cầu Đông Nam (X > 0)
            let mut val_x = 0.45f32;
            // Quét Carry Wave (chuỗi bit 1 liên tục từ phải qua) để tinh chỉnh vị trí
            let c = self.count_consecutive_ones(num);
            val_x += 0.01 * c as f32;
            val_x
        } else {
            // Số Chẵn: Bán cầu Tây Bắc (X < 0)
            let mut val_x = -0.55f32;
            // Điều chỉnh dựa trên số lần chia 2 liên tục (số bit 0 ở cuối)
            let z = num.trailing_zeros();
            val_x -= 0.01 * z as f32;
            val_x
        };

        // 2. Trục Y: Thanh ghi bit nhớ (Carry Wave)
        let y_raw = if num % 2 == 1 {
            // Số Lẻ: Y âm (Đông Nam)
            let c = self.count_consecutive_ones(num);
            let mut val_y = -0.85f32;
            // Áp lực bit nhớ đẩy tọa độ dịch chuyển dọc trục Y
            val_y += 0.01 * c as f32;
            val_y
        } else {
            // Số Chẵn: Y dương (Tây Bắc)
            let z = num.trailing_zeros();
            let mut val_y = 0.80f32;
            val_y -= 0.01 * z as f32;
            val_y
        };

        // 3. L2 Normalization: Ép tọa độ lên viền đường tròn đơn vị
        let len = (x_raw * x_raw + y_raw * y_raw).sqrt();
        if len > 0.0 {
            [x_raw / len, y_raw / len]
        } else {
            [1.0, 0.0]
        }
    }

    /// Khối Điều khiển Xung nhịp thích ứng: cấu hình góc xoay Theta
    pub fn dynamic_clock(&self, num: u64) -> f32 {
        // Kiểm tra xem có phải họ số Mersenne (2^k - 1)
        if num > 1 && (num + 1).is_power_of_two() {
            // Chế độ Xung vuông 2 pha (~180 độ)
            180.8f32.to_radians()
        } else {
            // Chế độ Đa pha (~160 độ)
            160.1f32.to_radians()
        }
    }

    /// Khối xoay ma trận 2D
    pub fn rotate(&self, coords: [f32; 2], angle: f32) -> [f32; 2] {
        let x = coords[0];
        let y = coords[1];
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        // Ma trận xoay hình học
        let new_x = x * cos_a + y * sin_a;
        let new_y = -x * sin_a + y * cos_a;
        [new_x, new_y]
    }

    /// Khối Vận hành Lệnh và Cập nhật trạng thái
    pub fn step(&mut self) -> (u64, [f32; 2], [f32; 2], f32) {
        let current_val = self.val;
        let current_latent = self.latent;
        let clock_angle = self.theta;

        // 1. Dự đoán tọa độ latent tiếp theo bằng phép xoay ma trận
        let predicted_latent = self.rotate(current_latent, clock_angle);

        // 2. Cập nhật dải băng theo luật Collatz
        let next_val = if current_val % 2 == 1 {
            current_val * 3 + 1
        } else {
            current_val / 2
        };

        // Cập nhật trạng thái của Chip
        self.val = next_val;
        self.latent = self.encode(next_val);
        self.theta = self.dynamic_clock(next_val);

        (next_val, predicted_latent, self.latent, clock_angle)
    }

    // Đếm số bit 1 liên tục từ phải sang trái (LSB trở lên)
    fn count_consecutive_ones(&self, mut num: u64) -> u32 {
        let mut count = 0;
        while num % 2 == 1 {
            count += 1;
            num /= 2;
        }
        count
    }
}

fn main() {
    println!("============================================================");
    println!("===     EMULATOR CON CHIP ẢO COLLATZ (VIRTUAL CHIP)      ===");
    println!("===   Mô phỏng động lực học hình học trên vòng tròn đơn vị  ===");
    println!("============================================================\n");

    let test_numbers = vec![12u64, 31u64];

    for &start in &test_numbers {
        println!("------------------------------------------------------------");
        let is_mersenne = (start + 1).is_power_of_two();
        println!(
            "CHẠY KIỂM CHỨNG VỚI SỐ BẮT ĐẦU: {} ({})",
            start,
            if is_mersenne { "Họ số Mersenne - Toàn bit 1" } else { "Số ngẫu nhiên" }
        );
        println!("------------------------------------------------------------");

        let mut chip = CollatzVirtualChip::new(start);

        for step_idx in 0..20 {
            let current_val = chip.val;
            let current_coords = chip.latent;
            
            // Thực hiện một bước xung nhịp
            let (next_val, predicted_coords, actual_coords, clock_angle) = chip.step();
            
            // Sai số dự đoán giữa xoay ma trận và encode thực tế
            let error = ((predicted_coords[0] - actual_coords[0]).powi(2) 
                + (predicted_coords[1] - actual_coords[1]).powi(2))
                .sqrt();

            let clock_mode = if clock_angle.to_degrees() > 170.0 { "2-Phase (180°)" } else { "Multi-Phase (160°)" };

            println!(
                "Step {:2}: n = {:4} -> {:4} | Clock: {:<17} ({:6.2}°) | Latent: ({:+.3}, {:+.3}) -> Pred: ({:+.3}, {:+.3}) vs Actual: ({:+.3}, {:+.3}) | Error: {:.6}",
                step_idx,
                current_val,
                next_val,
                clock_mode,
                clock_angle.to_degrees(),
                current_coords[0], current_coords[1],
                predicted_coords[0], predicted_coords[1],
                actual_coords[0], actual_coords[1],
                error
            );

            // Kiểm tra xem đã rơi vào vòng lặp 4 -> 2 -> 1 chưa
            if current_val == 4 && next_val == 2 {
                println!("  [INFO] Đã đi vào vòng lặp hạt nhân 4 -> 2 -> 1!");
            }
        }
        println!();
    }
}
