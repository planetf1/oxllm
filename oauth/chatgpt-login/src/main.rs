//! ChatGPT（OpenAI OAuth）登录与 token 助手。
//!
//! 能力由 octolib 的 `chatgpt` 提供方承担：
//! - `login`  —— Sign in with ChatGPT（OAuth2 + PKCE，loopback 回调），凭据写入
//!   `~/.config/octolib/chatgpt/auth.json`
//! - `token`  —— 触发 octolib 的到期自动刷新后输出 access token，供 oxllm 使用
//! - `models` —— 列出当前账号所在套餐可用的模型 slug（oxllm 里填这个）
//! - `status` —— 只看本地凭据状态，不发网络请求

use anyhow::{Context, Result};
use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn auth_path() -> Result<PathBuf> {
    Ok(dirs::config_dir()
        .context("无法确定配置目录")?
        .join("octolib")
        .join("chatgpt")
        .join("auth.json"))
}

#[derive(serde::Deserialize)]
struct AuthFile {
    access_token: String,
    refresh_token: String,
    client_id: String,
    #[serde(default)]
    expires_at: u64,
    #[serde(default)]
    email: Option<String>,
}

fn read_auth() -> Result<AuthFile> {
    let path = auth_path()?;
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("未找到 ChatGPT 凭据 {}，请先运行 login", path.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("凭据文件损坏：{}", path.display()))
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn human_left(expires_at: u64) -> String {
    let left = expires_at.saturating_sub(now());
    if left == 0 {
        "已过期（下次 `token` 会自动刷新）".to_string()
    } else if left < 3600 {
        format!("{left} 分钟后过期")
    } else if left < 86400 {
        format!("{} 小时后过期", left / 3600)
    } else {
        format!("{} 天后过期", left / 86400)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("help");
    match cmd {
        "login" => login().await,
        "token" => token(args.iter().any(|a| a == "--json")).await,
        "models" => models().await,
        "status" => status(),
        _ => {
            println!(
                "用法：oxllm-chatgpt-login <login|token|models|status>\n\
                 \n  login   浏览器授权（Sign in with ChatGPT），写入 auth.json\n\
                 \n  token   输出 access_token（临近过期时用 octolib 自动刷新）\n\
                 \n  token --json  以 JSON 输出 access_token / expires_at / email\n\
                 \n  models  列出当前套餐可用模型 slug\n\
                 \n  status  只读检查本地凭据"
            );
            Ok(())
        },
    }
}

async fn login() -> Result<()> {
    use octolib::llm::providers::chatgpt::start_login;

    let pending = start_login("oxllm").await?;
    println!("1) 在浏览器打开下面的地址并授权（需 ChatGPT Plus/Pro/Team/Enterprise）：\n");
    println!("{}\n", pending.authorize_url());
    println!(
        "2) 授权完成后浏览器会跳到 http://127.0.0.1:<端口>/callback —— \
         该回调必须落在**运行本命令的机器**上。"
    );
    println!("   在远端机器上运行时，先做端口转发（端口见上面 URL 里的 127.0.0.1:后数字），例如：");
    println!("   ssh -L <端口>:127.0.0.1:<端口> <user>@<host>\n");
    println!("等待回调中……");
    std::io::stdout().flush().ok();

    let account = pending.finish().await?;
    let auth = read_auth()?;
    println!(
        "登录成功：{}（{}）",
        account.email.clone().unwrap_or_else(|| "未知账号".into()),
        human_left(auth.expires_at)
    );
    println!("凭据已写入：{}", auth_path()?.display());
    println!("下一步：`oxllm-chatgpt-login models` 查看可用模型，再运行 oauth/sync-token.sh");
    Ok(())
}

async fn token(as_json: bool) -> Result<()> {
    // 先走一次 octolib 的带鉴权请求：临近过期或 401 时它会用 refresh_token 换新，
    // 并把新的 access_token 写回 auth.json（我们只是借用这个刷新副作用）。
    let probe = octolib::llm::providers::chatgpt::list_models().await;
    if let Err(err) = &probe {
        eprintln!("warn: 触发刷新时请求失败（继续使用本地凭据）：{err:#}");
    }

    let auth = read_auth()?;
    if as_json {
        println!(
            "{}",
            serde_json::json!({
                "access_token": auth.access_token,
                "expires_at": auth.expires_at,
                "email": auth.email,
            })
        );
    } else {
        // 只输出裸 token，便于脚本替换进 oxllm 配置
        println!("{}", auth.access_token);
        eprintln!(
            "info: {} · client_id {} · {}",
            human_left(auth.expires_at),
            auth.client_id,
            if auth.refresh_token.is_empty() {
                "无可刷新凭据"
            } else {
                "可自动刷新"
            }
        );
    }
    Ok(())
}

async fn models() -> Result<()> {
    let models = octolib::llm::providers::chatgpt::list_models().await?;
    if models.is_empty() {
        println!("该账号没有可用模型");
        return Ok(());
    }
    println!("{:<28} {}", "slug（oxllm 里填这个）", "显示名");
    for model in models {
        println!("{:<28} {}", model.slug, model.display_name);
    }
    Ok(())
}

fn status() -> Result<()> {
    let auth = read_auth()?;
    println!("凭据：{}", auth_path()?.display());
    println!("账号：{}", auth.email.unwrap_or_else(|| "未知".into()));
    println!("状态：{}", human_left(auth.expires_at));
    println!(
        "刷新：{}",
        if auth.refresh_token.is_empty() {
            "无 refresh_token，过期后需重新 login"
        } else {
            "有 refresh_token，`token` 会自动续期"
        }
    );
    Ok(())
}

// 保留：refresh_token / client_id 目前只用于打印诊断信息。
#[allow(dead_code)]
fn refresh_supported(auth: &AuthFile) -> bool {
    !auth.refresh_token.is_empty() && !auth.client_id.is_empty()
}
