class GLTFLoader {
  setMeshoptDecoder(_decoder) {
    // no-op for unit tests
  }
  load(_url, onLoad, _onProgress, _onError) {
    // Minimal GLTF stub for unit tests (no real parsing)
    onLoad({ scene: { traverse: () => {} } })
  }
}

module.exports = { GLTFLoader }
