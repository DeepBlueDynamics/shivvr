use anyhow::{anyhow, bail, Result};
use jsonwebtoken::{decode, Algorithm, DecodingKey, Validation};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NutsAuthClaims {
    pub user_id: String,
    pub email: String,
    pub name: Option<String>,
}

#[derive(Debug, Deserialize)]
struct JwksResponse {
    keys: Vec<JwkKey>,
}

#[derive(Debug, Deserialize)]
struct JwkKey {
    kty: String,
    #[serde(default)]
    alg: Option<String>,
    n: String,
    e: String,
}

#[derive(Debug, Deserialize)]
struct JwtClaims {
    user_id: String,
    sub: String,
    #[serde(default)]
    name: Option<String>,
    exp: usize,
}

#[derive(Debug, Serialize)]
struct ValidateApiTokenRequest<'a> {
    token: &'a str,
}

#[derive(Debug, Deserialize)]
struct ValidateApiTokenResponse {
    valid: bool,
    #[serde(default)]
    user_uid: Option<String>,
    #[serde(default)]
    subject: Option<String>,
    #[serde(default)]
    actor: Option<String>,
}

pub struct NutsAuth {
    jwks_url: String,
    validate_url: String,
    decoding_key: RwLock<Option<DecodingKey>>,
    client: Client,
}

impl NutsAuth {
    pub fn new(jwks_url: String, validate_url: String) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(15))
            .build()
            .expect("failed to construct nuts-auth reqwest client");

        Self {
            jwks_url,
            validate_url,
            decoding_key: RwLock::new(None),
            client,
        }
    }

    pub async fn refresh_jwks(&self) -> Result<()> {
        let response = self.client.get(&self.jwks_url).send().await?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            bail!("JWKS fetch failed {}: {}", status, body);
        }

        let jwks: JwksResponse = response.json().await?;
        let key = jwks
            .keys
            .into_iter()
            .find(|k| k.kty == "RSA" && k.alg.as_deref().unwrap_or("RS256") == "RS256")
            .ok_or_else(|| anyhow!("No RSA/RS256 key found in JWKS"))?;

        let decoding_key = DecodingKey::from_rsa_components(&key.n, &key.e)?;
        let mut cached = self.decoding_key.write().unwrap();
        *cached = Some(decoding_key);
        Ok(())
    }

    pub async fn verify_jwt(&self, token: &str) -> Result<NutsAuthClaims> {
        if self.decoding_key.read().unwrap().is_none() {
            self.refresh_jwks().await?;
        }

        let claims = match self.decode_jwt(token) {
            Ok(claims) => claims,
            Err(first_err) => {
                if self.decoding_key.read().unwrap().is_none() {
                    return Err(first_err);
                }
                Err(first_err)?
            }
        };

        Ok(NutsAuthClaims {
            user_id: claims.user_id,
            email: claims.sub,
            name: claims.name,
        })
    }

    pub async fn validate_api_token(&self, token: &str) -> Result<NutsAuthClaims> {
        let response = self
            .client
            .post(&self.validate_url)
            .json(&ValidateApiTokenRequest { token })
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            bail!("API token validation failed {}: {}", status, body);
        }

        let payload: ValidateApiTokenResponse = response.json().await?;
        if !payload.valid {
            bail!("invalid token");
        }

        Ok(NutsAuthClaims {
            user_id: payload
                .user_uid
                .ok_or_else(|| anyhow!("validate response missing user_uid"))?,
            email: payload
                .subject
                .ok_or_else(|| anyhow!("validate response missing subject"))?,
            name: payload.actor,
        })
    }

    pub async fn verify(&self, bearer: &str) -> Result<NutsAuthClaims> {
        if bearer.starts_with("ahp_") {
            self.validate_api_token(bearer).await
        } else {
            self.verify_jwt(bearer).await
        }
    }

    fn decode_jwt(&self, token: &str) -> Result<JwtClaims> {
        let cached = self.decoding_key.read().unwrap();
        let key = cached
            .as_ref()
            .ok_or_else(|| anyhow!("No JWKS decoding key loaded"))?;

        let mut validation = Validation::new(Algorithm::RS256);
        validation.validate_exp = true;

        let decoded = decode::<JwtClaims>(token, key, &validation)?;
        Ok(decoded.claims)
    }
}
