# Detailed Feature Comparison

**Research Date:** January 2025

## Protocol Support Comparison

| Product | VNC | RDP | Proprietary | SSH Tunnel | Tailscale |
|---------|-----|-----|-------------|------------|-----------|
| **Screens 5** | ✅ | ❌ | ❌ | ✅ Built-in | ✅ Built-in |
| **Jump Desktop** | ✅ | ✅ | ✅ Fluid | ✅ | ❌ |
| **Remotix** | ✅ | ✅ | ✅ NEAR | ✅ | ❌ |
| **TeamViewer** | ❌ | ❌ | ✅ | ❌ | ❌ |
| **AnyDesk** | ❌ | ❌ | ✅ | ❌ | ❌ |
| **Splashtop** | ❌ | ❌ | ✅ | ❌ | ❌ |
| **RealVNC** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Parsec** | ❌ | ❌ | ✅ | ❌ | ❌ |

**Key Insight:** Screens is unique in having built-in Tailscale integration, which is a significant advantage for security-conscious users.

---

## Platform Support Matrix

### Client Platforms (Connect FROM)

| Product | macOS | iOS | iPadOS | visionOS | Windows | Android | Linux | Web |
|---------|-------|-----|--------|----------|---------|---------|-------|-----|
| **Screens 5** | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Jump Desktop** | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ |
| **Remotix** | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ |
| **TeamViewer** | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| **AnyDesk** | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| **Splashtop** | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| **RealVNC** | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ |
| **Parsec** | ✅ | Limited | Limited | ❌ | ✅ | ✅ | ✅ | ❌ |
| **Chrome RD** | ✅* | ✅* | ✅* | ❌ | ✅* | ✅* | ✅* | ✅ |

*Via Chrome browser

### Host Platforms (Connect TO)

| Product | macOS | Windows | Linux | Raspberry Pi | iOS/iPadOS |
|---------|-------|---------|-------|--------------|------------|
| **Screens 5** | ✅ | ✅* | ✅* | ✅ | ❌ |
| **Jump Desktop** | ✅ | ✅ | ✅* | ✅ | ❌ |
| **Remotix** | ✅ | ✅ | ✅ | ✅ | ❌ |
| **TeamViewer** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **AnyDesk** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Splashtop** | ✅ | ✅ | ✅ | ❌ | ❌ |
| **RealVNC** | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Parsec** | ✅ | ✅ | ✅ | ✅ | ❌ |

*Requires third-party VNC server (e.g., TightVNC, UltraVNC)

**Key Insight:** Screens is the only product with native visionOS support, giving it first-mover advantage in the spatial computing market.

---

## Performance Features

| Feature | Screens 5 | Jump Desktop | Remotix | Splashtop | Parsec |
|---------|-----------|--------------|---------|-----------|--------|
| **Max Resolution** | 4K+ | 4K | 4K | 4K | 4K |
| **Max Frame Rate** | 60fps | 60fps | 60fps | 240fps | 60fps |
| **4:4:4 Color Mode** | ❌ | ❌ | ❌ | ✅ | ✅ |
| **Hardware Accel.** | ✅ | ✅ | ✅ (NEAR) | ✅ | ✅ |
| **Adaptive Quality** | ✅ | ✅ (Fluid) | ✅ (NEAR) | ✅ | ✅ |
| **Low Latency** | Good | Excellent | Excellent | Good | Excellent |
| **Audio Streaming** | ✅ | ✅ | ✅ | ✅ | ✅ |

**Key Insight:** Splashtop and Parsec lead in performance features, especially for creative professionals needing color accuracy.

---

## Security Features

| Feature | Screens 5 | Jump Desktop | TeamViewer | AnyDesk | RealVNC |
|---------|-----------|--------------|------------|---------|---------|
| **End-to-End Encryption** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Encryption Standard** | AES-256 | AES-256 | AES-256 | AES-256 | AES-256 |
| **2FA/MFA** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Touch ID/Face ID** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Apple Watch Unlock** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **SSO Support** | ❌ | ❌ | ✅ (Tensor) | ✅ (Ent) | ✅ |
| **Curtain/Privacy Mode** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Session Recording** | ❌ | ❌ | ✅ | ✅ | ✅ |
| **Audit Logs** | ❌ | ❌ | ✅ | ✅ | ✅ |
| **Self-Hosted Option** | ❌ | ❌ | ✅ (Tensor) | ✅ | ✅ |

**Key Insight:** Screens has unique Apple ecosystem security features (Apple Watch unlock) but lacks enterprise audit/logging features.

---

## Collaboration Features

| Feature | Screens 5 | Jump Desktop | TeamViewer | Splashtop | Parsec |
|---------|-----------|--------------|------------|-----------|--------|
| **Multiple Cursors** | ❌ | ✅ | ✅ | ❌ | ✅ |
| **Session Sharing** | ❌ | ✅ | ✅ | ❌ | ✅ |
| **In-Session Chat** | ❌ | ❌ | ✅ | ✅ (Pro) | ✅ |
| **Video Conferencing** | ❌ | ❌ | ✅ | ❌ | ❌ |
| **Whiteboard** | ❌ | ❌ | ✅ | ❌ | ❌ |
| **Guest Access** | Via Assist | ❌ | ✅ | ✅ | ✅ ($25) |

**Key Insight:** TeamViewer dominates collaboration features; Screens focuses on individual remote access.

---

## File Management

| Feature | Screens 5 | Jump Desktop | TeamViewer | Splashtop | RealVNC |
|---------|-----------|--------------|------------|-----------|---------|
| **File Transfer** | ✅ (Mac-Mac) | ✅ | ✅ | ✅ | ✅ |
| **Drag & Drop** | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Background Transfer** | ✅ | ❌ | ✅ | ❌ | ❌ |
| **Remote Printing** | ❌ | ❌ | ✅ | ✅ | ❌ |
| **Clipboard Sync** | ✅ | ✅ | ✅ | ✅ | ✅ |

**Key Insight:** Screens' background file transfer is a notable feature for large file workflows.

---

## Apple Ecosystem Integration

| Feature | Screens 5 | Jump Desktop | Remotix |
|---------|-----------|--------------|---------|
| **iCloud Sync** | ✅ | ✅ | ✅ |
| **Handoff** | ✅ | ❌ | ❌ |
| **Touch ID/Face ID** | ✅ | ✅ | ✅ |
| **Apple Watch Unlock** | ✅ | ❌ | ❌ |
| **Vision Pro Native** | ✅ | ❌ | ❌ |
| **Keyboard Shortcuts** | Native | Mapped | Mapped |
| **Trackpad Gestures** | ✅ | ✅ | ✅ |
| **External KB/Mouse** | ✅ | ✅ | ✅ |
| **iPad Stage Manager** | ✅ | ✅ | ❌ |
| **macOS Menus (iPad)** | ✅ | ❌ | ❌ |
| **Siri Shortcuts** | ✅ | ❌ | ❌ |

**Key Insight:** Screens has the deepest Apple ecosystem integration, especially with Vision Pro and Handoff support.

---

## Input Device Support

| Device Type | Screens 5 | Jump Desktop | Parsec | Splashtop |
|-------------|-----------|--------------|--------|-----------|
| **Keyboard (External)** | ✅ | ✅ | ✅ | ✅ |
| **Mouse** | ✅ | ✅ | ✅ | ✅ |
| **Trackpad** | ✅ | ✅ | ✅ | ✅ |
| **Gamepad** | ❌ | ❌ | ✅ | ❌ |
| **Drawing Tablet** | Limited | Limited | ✅ | ✅ (Wacom Bridge) |
| **Stylus (Pen)** | Basic | Basic | ✅ | ✅ |

**Key Insight:** Parsec and Splashtop excel for creative professionals needing drawing tablet support.

---

## Connection & Discovery

| Feature | Screens 5 | Jump Desktop | TeamViewer | AnyDesk |
|---------|-----------|--------------|------------|---------|
| **Local Discovery** | ✅ | ✅ | ✅ | ✅ |
| **Cloud Relay** | ✅ (Screens Connect) | ✅ | ✅ | ✅ |
| **Direct Connect (IP)** | ✅ | ✅ | ✅ | ✅ |
| **QR Code Setup** | ❌ | ❌ | ✅ | ✅ |
| **Wake on LAN** | ✅ | ✅ | ✅ | ✅ |
| **Connection Speed** | Fast | Slower | Fast | Fast |
| **Screen Update Speed** | Good | Excellent (Fluid) | Good | Good |

**Key Insight:** Jump Desktop connects slightly slower but has faster screen updates with Fluid protocol.

---

## Helper Utilities

| Product | Helper Utility | Purpose | Platform |
|---------|----------------|---------|----------|
| **Screens 5** | Screens Connect | Easy remote access setup | Mac, Windows |
| **Screens 5** | Screens Assist | One-click family support | Mac |
| **Jump Desktop** | Jump Desktop Connect | Remote access setup | Mac, Windows |
| **TeamViewer** | QuickSupport | Ad-hoc support sessions | All |
| **Splashtop** | Splashtop Streamer | Host installation | All |
| **AnyDesk** | AnyDesk Portable | No-install client | Windows |

---

## Mobile Experience Comparison

### iPad-Specific Features

| Feature | Screens 5 | Jump Desktop | TeamViewer | Splashtop |
|---------|-----------|--------------|------------|-----------|
| **Multi-Window (iPadOS)** | ✅ | ✅ | ✅ | ✅ |
| **Stage Manager** | ✅ | ✅ | ❌ | ❌ |
| **Keyboard Shortcuts** | ✅ | ✅ | ✅ | ✅ |
| **Mouse Support** | ✅ | ✅ | ✅ | ✅ |
| **Full Screen Mode** | ✅ | ✅ | ✅ | ✅ |
| **Gesture Controls** | Native | Good | Basic | Basic |

### Vision Pro Experience

| Feature | Screens 5 | Others |
|---------|-----------|--------|
| **Native visionOS App** | ✅ | ❌ |
| **Hand Gesture Control** | ✅ | N/A |
| **Virtual Display Size** | Adjustable | N/A |
| **Eye Tracking Navigation** | ✅ | N/A |
| **Spatial Positioning** | ✅ | N/A |

**Key Insight:** Screens 5 is currently the only major VNC client with a native visionOS app, giving it significant first-mover advantage.

---

## Enterprise & Admin Features

| Feature | Screens 5 | TeamViewer | AnyDesk | Splashtop |
|---------|-----------|------------|---------|-----------|
| **User Management** | Basic (Org) | ✅ | ✅ | ✅ |
| **Group Policies** | ❌ | ✅ | ✅ | ✅ |
| **Mass Deployment** | ❌ | ✅ (MSI) | ✅ | ✅ |
| **Custom Branding** | ❌ | ✅ | ✅ | ✅ |
| **API Access** | ❌ | ✅ | ✅ | ✅ |
| **Integrations** | Tailscale | Many | Many | Many |
| **Compliance Certs** | ❌ | SOC2, ISO | SOC2 | SOC2, HIPAA |

**Key Insight:** Screens is not positioned for enterprise IT management; it's a prosumer/individual tool.

---

## Summary: Who Wins Each Category

| Category | Winner | Runner-Up |
|----------|--------|-----------|
| **Apple Integration** | Screens 5 | Jump Desktop |
| **Vision Pro** | Screens 5 | (No competition) |
| **Performance** | Parsec / Jump Desktop (Fluid) | Remotix (NEAR) |
| **Color Accuracy** | Splashtop | Parsec |
| **Enterprise Features** | TeamViewer | AnyDesk |
| **Value (Paid)** | Jump Desktop | Screens 5 |
| **Value (Free)** | RustDesk | Parsec Free |
| **Security** | RealVNC / TeamViewer | Screens 5 |
| **Creative Professionals** | Splashtop | Parsec |
| **Gaming** | Parsec | Jump Desktop |
| **Cross-Platform** | TeamViewer | AnyDesk |

---

*Feature data collected January 2025. Verify current capabilities before purchase decisions.*
