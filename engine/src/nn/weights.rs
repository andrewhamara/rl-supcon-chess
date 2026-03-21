use std::io::{self, Read, Write};
use std::fs::File;
use std::path::Path;

/// Weight storage for a single residual block.
#[derive(Clone)]
pub struct ResidualBlockWeights {
    pub conv1_weight: Vec<f32>, // [out_c, in_c, 3, 3]
    pub conv1_bias: Vec<f32>,   // [out_c]
    pub bn1_weight: Vec<f32>,   // [out_c] (gamma)
    pub bn1_bias: Vec<f32>,     // [out_c] (beta)
    pub bn1_mean: Vec<f32>,     // [out_c] (running mean)
    pub bn1_var: Vec<f32>,      // [out_c] (running var)

    pub conv2_weight: Vec<f32>,
    pub conv2_bias: Vec<f32>,
    pub bn2_weight: Vec<f32>,
    pub bn2_bias: Vec<f32>,
    pub bn2_mean: Vec<f32>,
    pub bn2_var: Vec<f32>,
}

/// Complete network weights.
#[derive(Clone)]
pub struct NetworkWeights {
    pub num_blocks: usize,
    pub num_filters: usize,

    // Stem
    pub stem_conv_weight: Vec<f32>, // [num_filters, 21, 3, 3]
    pub stem_conv_bias: Vec<f32>,
    pub stem_bn_weight: Vec<f32>,
    pub stem_bn_bias: Vec<f32>,
    pub stem_bn_mean: Vec<f32>,
    pub stem_bn_var: Vec<f32>,

    // Residual tower
    pub residual_blocks: Vec<ResidualBlockWeights>,

    // Policy head
    pub policy_conv_weight: Vec<f32>, // [32, num_filters, 1, 1]
    pub policy_conv_bias: Vec<f32>,
    pub policy_bn_weight: Vec<f32>,
    pub policy_bn_bias: Vec<f32>,
    pub policy_bn_mean: Vec<f32>,
    pub policy_bn_var: Vec<f32>,
    pub policy_fc_weight: Vec<f32>, // [1858, 2048]
    pub policy_fc_bias: Vec<f32>,

    // Value head
    pub value_conv_weight: Vec<f32>, // [1, num_filters, 1, 1]
    pub value_conv_bias: Vec<f32>,
    pub value_bn_weight: Vec<f32>,
    pub value_bn_bias: Vec<f32>,
    pub value_bn_mean: Vec<f32>,
    pub value_bn_var: Vec<f32>,
    pub value_fc1_weight: Vec<f32>, // [128, 64]
    pub value_fc1_bias: Vec<f32>,
    pub value_fc2_weight: Vec<f32>, // [1, 128]
    pub value_fc2_bias: Vec<f32>,
}

const MAGIC: u32 = 0xCE55_0001;
const VERSION: u32 = 1;

impl NetworkWeights {
    /// Create random weights for testing.
    pub fn random(num_blocks: usize, num_filters: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let mut rand_vec = |size: usize| -> Vec<f32> {
            let scale = (2.0 / size as f32).sqrt(); // He initialization
            (0..size).map(|_| rng.gen::<f32>() * scale - scale / 2.0).collect()
        };
        let zeros = |size: usize| -> Vec<f32> { vec![0.0; size] };
        let ones = |size: usize| -> Vec<f32> { vec![1.0; size] };

        let nf = num_filters;
        let input_planes = 21;

        let residual_blocks = (0..num_blocks)
            .map(|_| ResidualBlockWeights {
                conv1_weight: rand_vec(nf * nf * 3 * 3),
                conv1_bias: zeros(nf),
                bn1_weight: ones(nf),
                bn1_bias: zeros(nf),
                bn1_mean: zeros(nf),
                bn1_var: ones(nf),
                conv2_weight: rand_vec(nf * nf * 3 * 3),
                conv2_bias: zeros(nf),
                bn2_weight: ones(nf),
                bn2_bias: zeros(nf),
                bn2_mean: zeros(nf),
                bn2_var: ones(nf),
            })
            .collect();

        NetworkWeights {
            num_blocks,
            num_filters: nf,
            stem_conv_weight: rand_vec(nf * input_planes * 3 * 3),
            stem_conv_bias: zeros(nf),
            stem_bn_weight: ones(nf),
            stem_bn_bias: zeros(nf),
            stem_bn_mean: zeros(nf),
            stem_bn_var: ones(nf),
            residual_blocks,
            policy_conv_weight: rand_vec(32 * nf * 1 * 1),
            policy_conv_bias: zeros(32),
            policy_bn_weight: ones(32),
            policy_bn_bias: zeros(32),
            policy_bn_mean: zeros(32),
            policy_bn_var: ones(32),
            policy_fc_weight: rand_vec(1858 * 2048),
            policy_fc_bias: zeros(1858),
            value_conv_weight: rand_vec(1 * nf * 1 * 1),
            value_conv_bias: zeros(1),
            value_bn_weight: ones(1),
            value_bn_bias: zeros(1),
            value_bn_mean: zeros(1),
            value_bn_var: ones(1),
            value_fc1_weight: rand_vec(128 * 64),
            value_fc1_bias: zeros(128),
            value_fc2_weight: rand_vec(1 * 128),
            value_fc2_bias: zeros(1),
        }
    }

    /// Save weights to binary file.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let mut f = File::create(path)?;

        // Header
        f.write_all(&MAGIC.to_le_bytes())?;
        f.write_all(&VERSION.to_le_bytes())?;
        f.write_all(&(self.num_blocks as u32).to_le_bytes())?;
        f.write_all(&(self.num_filters as u32).to_le_bytes())?;

        // Write all weight tensors
        write_tensor(&mut f, &self.stem_conv_weight)?;
        write_tensor(&mut f, &self.stem_conv_bias)?;
        write_tensor(&mut f, &self.stem_bn_weight)?;
        write_tensor(&mut f, &self.stem_bn_bias)?;
        write_tensor(&mut f, &self.stem_bn_mean)?;
        write_tensor(&mut f, &self.stem_bn_var)?;

        for block in &self.residual_blocks {
            write_tensor(&mut f, &block.conv1_weight)?;
            write_tensor(&mut f, &block.conv1_bias)?;
            write_tensor(&mut f, &block.bn1_weight)?;
            write_tensor(&mut f, &block.bn1_bias)?;
            write_tensor(&mut f, &block.bn1_mean)?;
            write_tensor(&mut f, &block.bn1_var)?;
            write_tensor(&mut f, &block.conv2_weight)?;
            write_tensor(&mut f, &block.conv2_bias)?;
            write_tensor(&mut f, &block.bn2_weight)?;
            write_tensor(&mut f, &block.bn2_bias)?;
            write_tensor(&mut f, &block.bn2_mean)?;
            write_tensor(&mut f, &block.bn2_var)?;
        }

        write_tensor(&mut f, &self.policy_conv_weight)?;
        write_tensor(&mut f, &self.policy_conv_bias)?;
        write_tensor(&mut f, &self.policy_bn_weight)?;
        write_tensor(&mut f, &self.policy_bn_bias)?;
        write_tensor(&mut f, &self.policy_bn_mean)?;
        write_tensor(&mut f, &self.policy_bn_var)?;
        write_tensor(&mut f, &self.policy_fc_weight)?;
        write_tensor(&mut f, &self.policy_fc_bias)?;

        write_tensor(&mut f, &self.value_conv_weight)?;
        write_tensor(&mut f, &self.value_conv_bias)?;
        write_tensor(&mut f, &self.value_bn_weight)?;
        write_tensor(&mut f, &self.value_bn_bias)?;
        write_tensor(&mut f, &self.value_bn_mean)?;
        write_tensor(&mut f, &self.value_bn_var)?;
        write_tensor(&mut f, &self.value_fc1_weight)?;
        write_tensor(&mut f, &self.value_fc1_bias)?;
        write_tensor(&mut f, &self.value_fc2_weight)?;
        write_tensor(&mut f, &self.value_fc2_bias)?;

        Ok(())
    }

    /// Load weights from binary file.
    pub fn load(path: &Path) -> io::Result<Self> {
        let mut f = File::open(path)?;

        let magic = read_u32(&mut f)?;
        if magic != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid weight file magic"));
        }
        let version = read_u32(&mut f)?;
        if version != VERSION {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Unsupported weight file version"));
        }
        let num_blocks = read_u32(&mut f)? as usize;
        let num_filters = read_u32(&mut f)? as usize;

        let stem_conv_weight = read_tensor(&mut f)?;
        let stem_conv_bias = read_tensor(&mut f)?;
        let stem_bn_weight = read_tensor(&mut f)?;
        let stem_bn_bias = read_tensor(&mut f)?;
        let stem_bn_mean = read_tensor(&mut f)?;
        let stem_bn_var = read_tensor(&mut f)?;

        let mut residual_blocks = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            residual_blocks.push(ResidualBlockWeights {
                conv1_weight: read_tensor(&mut f)?,
                conv1_bias: read_tensor(&mut f)?,
                bn1_weight: read_tensor(&mut f)?,
                bn1_bias: read_tensor(&mut f)?,
                bn1_mean: read_tensor(&mut f)?,
                bn1_var: read_tensor(&mut f)?,
                conv2_weight: read_tensor(&mut f)?,
                conv2_bias: read_tensor(&mut f)?,
                bn2_weight: read_tensor(&mut f)?,
                bn2_bias: read_tensor(&mut f)?,
                bn2_mean: read_tensor(&mut f)?,
                bn2_var: read_tensor(&mut f)?,
            });
        }

        let policy_conv_weight = read_tensor(&mut f)?;
        let policy_conv_bias = read_tensor(&mut f)?;
        let policy_bn_weight = read_tensor(&mut f)?;
        let policy_bn_bias = read_tensor(&mut f)?;
        let policy_bn_mean = read_tensor(&mut f)?;
        let policy_bn_var = read_tensor(&mut f)?;
        let policy_fc_weight = read_tensor(&mut f)?;
        let policy_fc_bias = read_tensor(&mut f)?;

        let value_conv_weight = read_tensor(&mut f)?;
        let value_conv_bias = read_tensor(&mut f)?;
        let value_bn_weight = read_tensor(&mut f)?;
        let value_bn_bias = read_tensor(&mut f)?;
        let value_bn_mean = read_tensor(&mut f)?;
        let value_bn_var = read_tensor(&mut f)?;
        let value_fc1_weight = read_tensor(&mut f)?;
        let value_fc1_bias = read_tensor(&mut f)?;
        let value_fc2_weight = read_tensor(&mut f)?;
        let value_fc2_bias = read_tensor(&mut f)?;

        Ok(NetworkWeights {
            num_blocks,
            num_filters,
            stem_conv_weight, stem_conv_bias,
            stem_bn_weight, stem_bn_bias, stem_bn_mean, stem_bn_var,
            residual_blocks,
            policy_conv_weight, policy_conv_bias,
            policy_bn_weight, policy_bn_bias, policy_bn_mean, policy_bn_var,
            policy_fc_weight, policy_fc_bias,
            value_conv_weight, value_conv_bias,
            value_bn_weight, value_bn_bias, value_bn_mean, value_bn_var,
            value_fc1_weight, value_fc1_bias,
            value_fc2_weight, value_fc2_bias,
        })
    }
}

fn write_tensor(f: &mut File, data: &[f32]) -> io::Result<()> {
    let len = data.len() as u32;
    f.write_all(&len.to_le_bytes())?;
    for &v in data {
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn read_tensor(f: &mut File) -> io::Result<Vec<f32>> {
    let len = read_u32(f)? as usize;
    let mut data = vec![0.0f32; len];
    for v in &mut data {
        let mut buf = [0u8; 4];
        f.read_exact(&mut buf)?;
        *v = f32::from_le_bytes(buf);
    }
    Ok(data)
}

fn read_u32(f: &mut File) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}
