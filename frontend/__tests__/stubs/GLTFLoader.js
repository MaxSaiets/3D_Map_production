class GLTFLoader {
  load(_url, onLoad, _onProgress, _onError) {
    // Minimal GLTF stub for unit tests (no real parsing)
    onLoad({ scene: { traverse: () => {} } })
  }
}

module.exports = { GLTFLoader }
