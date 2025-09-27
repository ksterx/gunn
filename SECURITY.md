# Security Policy

## Supported Versions

We provide security updates for the following versions of gunn:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security vulnerability in gunn, please report it responsibly by following these steps:

### Private Disclosure

**DO NOT** create a public GitHub issue for security vulnerabilities. Instead, please:

1. **Email**: Send details to [security@your-domain.com] (replace with actual contact)
2. **Subject**: Include "SECURITY" in the subject line
3. **Details**: Provide as much information as possible:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### What to Expect

- **Acknowledgment**: We will acknowledge receipt within 48 hours
- **Initial Assessment**: We will provide an initial assessment within 5 business days
- **Updates**: We will keep you informed of our progress
- **Resolution**: We aim to resolve critical vulnerabilities within 30 days
- **Credit**: We will credit you in our security advisory (unless you prefer to remain anonymous)

### Security Considerations in gunn

The gunn multi-agent simulation system handles potentially sensitive data and provides multi-tenant capabilities. Key security areas include:

#### Authentication and Authorization
- Agent identity verification
- Tenant isolation enforcement
- API access controls
- Rate limiting and quota enforcement

#### Data Protection
- PII redaction in logs and telemetry
- Secure storage of event logs
- Encrypted communication channels
- Memory protection for sensitive data

#### System Security
- Input validation and sanitization
- SQL injection prevention
- Cross-tenant data leakage prevention
- Secure configuration management

#### Network Security
- TLS/mTLS for external communications
- WebSocket security
- API endpoint protection
- Network isolation between tenants

### Security Best Practices for Users

When deploying gunn in production:

1. **Authentication**: Always enable authentication for external adapters
2. **Network Security**: Use TLS for all external communications
3. **Monitoring**: Enable security event logging and monitoring
4. **Updates**: Keep gunn and dependencies updated
5. **Configuration**: Follow security configuration guidelines
6. **Isolation**: Properly configure tenant isolation
7. **Logging**: Enable PII redaction in production logs

### Vulnerability Categories

We consider the following types of vulnerabilities:

#### Critical
- Remote code execution
- Authentication bypass
- Cross-tenant data access
- Privilege escalation

#### High
- Information disclosure
- Denial of service
- Data corruption
- Session hijacking

#### Medium
- Cross-site scripting (XSS)
- Cross-site request forgery (CSRF)
- Information leakage
- Weak cryptography

#### Low
- Security misconfigurations
- Weak password policies
- Information disclosure (limited)

### Security Updates

Security updates will be:
- Released as patch versions (e.g., 0.1.1 → 0.1.2)
- Documented in release notes
- Announced through appropriate channels
- Backported to supported versions when possible

### Bug Bounty Program

We do not currently have a formal bug bounty program, but we appreciate responsible disclosure and will acknowledge security researchers who help improve gunn's security.

### Contact Information

For security-related questions or concerns:
- **Security Email**: [security@your-domain.com]
- **General Contact**: [contact@your-domain.com]
- **GitHub**: Create a private security advisory

### Legal

By reporting security vulnerabilities, you agree to:
- Allow us reasonable time to investigate and fix the issue
- Not publicly disclose the vulnerability until we have addressed it
- Not access or modify data beyond what is necessary to demonstrate the vulnerability

We commit to:
- Respond to your report in a timely manner
- Keep you informed of our progress
- Credit you appropriately (if desired)
- Not pursue legal action against researchers who follow responsible disclosure
