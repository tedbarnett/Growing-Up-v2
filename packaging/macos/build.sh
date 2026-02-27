#!/usr/bin/env bash
#
# build.sh — Build Growing Up.app, create DMG, optionally sign + notarize.
#
# Usage:
#   ./packaging/macos/build.sh              # Build without code signing
#   ./packaging/macos/build.sh --sign       # Build + code sign
#   ./packaging/macos/build.sh --notarize   # Build + sign + notarize
#
# Prerequisites:
#   pip install pyinstaller   (in the project venv)
#
# For distribution (--sign / --notarize), you need:
#   - A "Developer ID Application" certificate from developer.apple.com
#   - Set DEVELOPER_ID below or export it as an env var
#   - For notarization: set APPLE_ID, TEAM_ID, and create an app-specific
#     password stored in the keychain as "notarytool-password"
#
set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PACKAGING_DIR="$SCRIPT_DIR"
DIST_DIR="$PROJECT_ROOT/dist"
BUILD_DIR="$PROJECT_ROOT/build"
BIN_DIR="$PACKAGING_DIR/bin"

APP_NAME="Growing Up"
DMG_NAME="Growing Up.dmg"
VOLUME_NAME="Growing Up"

# Code signing identity — set this to your Developer ID Application cert.
# You can also export DEVELOPER_ID="Developer ID Application: ..." before running.
DEVELOPER_ID="${DEVELOPER_ID:-}"

# Notarization credentials
APPLE_ID="${APPLE_ID:-info@barnettlabs.com}"
TEAM_ID="${TEAM_ID:-}"

# Parse flags
DO_SIGN=false
DO_NOTARIZE=false
for arg in "$@"; do
    case "$arg" in
        --sign) DO_SIGN=true ;;
        --notarize) DO_NOTARIZE=true; DO_SIGN=true ;;
    esac
done

# ─── Step 1: Generate icon.icns from the baby photo ──────────────────────────

echo "==> Generating app icon..."

ICON_SOURCE="$PROJECT_ROOT/webapp/static/apple-touch-icon.png"
ICONSET_DIR="$PACKAGING_DIR/GrowingUp.iconset"
ICNS_FILE="$PACKAGING_DIR/icon.icns"

if [ ! -f "$ICNS_FILE" ]; then
    if [ ! -f "$ICON_SOURCE" ]; then
        echo "WARNING: No icon source at $ICON_SOURCE — using default icon"
    else
        mkdir -p "$ICONSET_DIR"

        # Generate all required sizes
        for size in 16 32 64 128 256 512; do
            sips -z $size $size "$ICON_SOURCE" --out "$ICONSET_DIR/icon_${size}x${size}.png" >/dev/null 2>&1
        done
        # Retina versions
        for size in 32 64 128 256 512 1024; do
            half=$((size / 2))
            sips -z $size $size "$ICON_SOURCE" --out "$ICONSET_DIR/icon_${half}x${half}@2x.png" >/dev/null 2>&1
        done

        iconutil -c icns "$ICONSET_DIR" -o "$ICNS_FILE"
        rm -rf "$ICONSET_DIR"
        echo "    Created icon.icns"
    fi
else
    echo "    icon.icns already exists, skipping"
fi

# ─── Step 2: Download static ffmpeg + ffprobe if not cached ──────────────────

echo "==> Checking for bundled ffmpeg..."

mkdir -p "$BIN_DIR"

if [ ! -f "$BIN_DIR/ffmpeg" ] || [ ! -f "$BIN_DIR/ffprobe" ]; then
    echo "    Downloading static ffmpeg for macOS arm64..."

    FFMPEG_URL="https://evermeet.cx/ffmpeg/getrelease/ffmpeg/zip"
    FFPROBE_URL="https://evermeet.cx/ffmpeg/getrelease/ffprobe/zip"

    TMPDIR_DL="$(mktemp -d)"

    # Download ffmpeg
    echo "    Downloading ffmpeg..."
    curl -L -o "$TMPDIR_DL/ffmpeg.zip" "$FFMPEG_URL"
    unzip -o -q "$TMPDIR_DL/ffmpeg.zip" -d "$BIN_DIR"

    # Download ffprobe
    echo "    Downloading ffprobe..."
    curl -L -o "$TMPDIR_DL/ffprobe.zip" "$FFPROBE_URL"
    unzip -o -q "$TMPDIR_DL/ffprobe.zip" -d "$BIN_DIR"

    chmod +x "$BIN_DIR/ffmpeg" "$BIN_DIR/ffprobe"
    rm -rf "$TMPDIR_DL"
    echo "    ffmpeg and ffprobe ready"
else
    echo "    ffmpeg and ffprobe already cached"
fi

# ─── Step 3: Run PyInstaller ─────────────────────────────────────────────────

echo "==> Running PyInstaller..."

# Activate venv if available
if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
fi

# Ensure PyInstaller is installed
if ! command -v pyinstaller &>/dev/null; then
    echo "    Installing PyInstaller..."
    pip install pyinstaller
fi

# Clean previous builds
rm -rf "$DIST_DIR/$APP_NAME" "$DIST_DIR/$APP_NAME.app"

pyinstaller \
    --distpath "$DIST_DIR" \
    --workpath "$BUILD_DIR" \
    --noconfirm \
    "$PACKAGING_DIR/GrowingUp.spec"

echo "    Built $APP_NAME.app"

# ─── Step 4: Code sign (optional) ────────────────────────────────────────────

APP_PATH="$DIST_DIR/$APP_NAME.app"

if $DO_SIGN; then
    if [ -z "$DEVELOPER_ID" ]; then
        echo ""
        echo "ERROR: --sign requires DEVELOPER_ID to be set."
        echo "  export DEVELOPER_ID=\"Developer ID Application: Your Name (TEAMID)\""
        echo ""
        echo "Available identities:"
        security find-identity -v -p codesigning | head -10
        echo ""
        echo "For distribution you need a 'Developer ID Application' certificate."
        echo "Create one at: https://developer.apple.com/account/resources/certificates"
        exit 1
    fi

    echo "==> Code signing with: $DEVELOPER_ID"

    # Sign all .so, .dylib, and framework files first, then the app itself
    find "$APP_PATH" -type f \( -name "*.so" -o -name "*.dylib" \) -exec \
        codesign --force --sign "$DEVELOPER_ID" \
        --entitlements "$PACKAGING_DIR/entitlements.plist" \
        --timestamp --options runtime {} \;

    codesign --deep --force --sign "$DEVELOPER_ID" \
        --entitlements "$PACKAGING_DIR/entitlements.plist" \
        --timestamp --options runtime \
        "$APP_PATH"

    echo "    Verifying signature..."
    codesign --verify --verbose "$APP_PATH"
    echo "    Code signing complete"
fi

# ─── Step 5: Create DMG ──────────────────────────────────────────────────────

echo "==> Creating DMG..."

DMG_PATH="$DIST_DIR/$DMG_NAME"
DMG_TEMP="$DIST_DIR/tmp_$DMG_NAME"

# Clean up any previous DMG
rm -f "$DMG_PATH" "$DMG_TEMP"

# Create a temporary DMG
hdiutil create \
    -volname "$VOLUME_NAME" \
    -srcfolder "$APP_PATH" \
    -ov -format UDRW \
    "$DMG_TEMP"

# Mount it to customize
MOUNT_POINT="/Volumes/$VOLUME_NAME"

# Unmount if already mounted
if [ -d "$MOUNT_POINT" ]; then
    hdiutil detach "$MOUNT_POINT" -force 2>/dev/null || true
fi

hdiutil attach "$DMG_TEMP" -mountpoint "$MOUNT_POINT"

# Add Applications symlink for drag-and-drop install
ln -sf /Applications "$MOUNT_POINT/Applications"

# Set DMG window appearance via AppleScript
osascript <<APPLESCRIPT
tell application "Finder"
    tell disk "$VOLUME_NAME"
        open
        set current view of container window to icon view
        set toolbar visible of container window to false
        set statusbar visible of container window to false
        set bounds of container window to {200, 120, 700, 400}
        set theViewOptions to the icon view options of container window
        set arrangement of theViewOptions to not arranged
        set icon size of theViewOptions to 80
        set position of item "$APP_NAME.app" of container window to {130, 150}
        set position of item "Applications" of container window to {370, 150}
        close
        open
        update without registering applications
    end tell
end tell
APPLESCRIPT

# Wait for Finder
sleep 2

# Unmount
hdiutil detach "$MOUNT_POINT"

# Convert to compressed read-only DMG
hdiutil convert "$DMG_TEMP" -format UDZO -imagekey zlib-level=9 -o "$DMG_PATH"
rm -f "$DMG_TEMP"

echo "    Created $DMG_PATH"

# ─── Step 6: Notarize (optional) ─────────────────────────────────────────────

if $DO_NOTARIZE; then
    if [ -z "$TEAM_ID" ]; then
        echo "ERROR: --notarize requires TEAM_ID to be set"
        exit 1
    fi

    echo "==> Submitting for notarization..."
    echo "    This may take several minutes..."

    xcrun notarytool submit "$DMG_PATH" \
        --apple-id "$APPLE_ID" \
        --team-id "$TEAM_ID" \
        --keychain-profile "notarytool-password" \
        --wait

    echo "==> Stapling notarization ticket..."
    xcrun stapler staple "$DMG_PATH"

    echo "    Notarization complete"
fi

# ─── Done ─────────────────────────────────────────────────────────────────────

echo ""
echo "=========================================="
echo "  Build complete!"
echo "=========================================="
echo "  App:  $APP_PATH"
echo "  DMG:  $DMG_PATH"
echo ""
if ! $DO_SIGN; then
    echo "  NOTE: App is not code signed. For distribution, run:"
    echo "    DEVELOPER_ID=\"Developer ID Application: ...\" ./packaging/macos/build.sh --sign"
fi
echo ""
