import axios from "axios";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ── Auth token provider ───────────────────────────────────────────────────────
type TokenProvider = (() => Promise<string | null>) | null;
let _tokenProvider: TokenProvider = null;

export function setApiAuthTokenProvider(provider: TokenProvider) {
  _tokenProvider = provider;
}

// Guard against mocked/partial axios in unit tests (auto-mock has no interceptors).
// In production real axios always exposes interceptors, so this is a no-op there.
if (axios?.interceptors?.request) {
  axios.interceptors.request.use(async (config) => {
    if (_tokenProvider) {
      try {
        const token = await _tokenProvider();
        if (token) {
          config.headers = config.headers ?? {};
          config.headers["Authorization"] = `Bearer ${token}`;
        }
      } catch {
        // ignore token errors
      }
    }
    return config;
  });
}

// ── Account types ─────────────────────────────────────────────────────────────
export interface AccountModel {
  id: string;
  title?: string;
  city?: string;
  status: string;
  progress?: number;
  model_size_mm?: number;
  created_at?: string;
  error?: string;
  message?: string;
  download_url?: string | null;
  download_url_3mf?: string | null;
  download_url_stl?: string | null;
  layers?: {
    terrain?: boolean;
    roads?: boolean;
    buildings?: boolean;
    water?: boolean;
    parks?: boolean;
  };
  material?: string;
  preview_snapshot?: any;
}

export interface AccountResponse {
  models: AccountModel[];
  usage?: {
    remaining: number;
    free_limit: number;
    completed: number;
    used: number;
  };
}

// ── Fast preview types ────────────────────────────────────────────────────────
export interface FastPreviewResponse {
  layers: {
    terrain: {
      heightfield?: {
        x: number[];
        y: number[];
        z: number[][];
        z_max_m?: number;
        [key: string]: any;
      } | null;
    };
    buildings: {
      features?: any[];
    };
    roads: {
      features?: any[];
    };
    water: {
      features?: any[];
    };
    parks: {
      features?: any[];
    };
  };
  bounds: {
    north: number;
    south: number;
    east: number;
    west: number;
  };
  center: {
    lat: number;
    lng: number;
  };
  model_logic?: {
    model_size_mm?: number;
    scale_factor_mm_per_m?: number;
    terrain_base_thickness_mm?: number;
    terrain_z_scale?: number;
    road_height_mm?: number;
    parks_height_mm?: number;
    preview_message?: string;
    [key: string]: any;
  };
  preview_id?: string;
  preview_status?: string;
  preview_stl?: string | null;
  model_file_url?: string | null;
  material?: string;
}

export interface GenerationRequest {
  north: number;
  south: number;
  east: number;
  west: number;
  road_width_multiplier: number;
  road_height_mm: number;
  road_embed_mm: number;
  building_min_height: number;
  building_height_multiplier: number;
  building_foundation_mm: number;
  building_embed_mm: number;
  water_depth: number;
  terrain_enabled: boolean;
  terrain_z_scale: number;
  terrain_base_thickness_mm: number;
  terrain_resolution: number;
  terrarium_zoom: number;
  flatten_buildings_on_terrain?: boolean;
  export_format: "stl" | "3mf";
  model_size_mm: number;
  context_padding_m?: number;
  is_ams_mode: boolean;
  flat_plate_mode?: boolean;
  flat_water_layer_mm?: number;
  flat_roads_layer_mm?: number;
  flat_parks_layer_mm?: number;
  flat_max_building_height_mm?: number;
  flat_uniform_building_height?: boolean;
  magnet_pocket?: boolean;
  magnet_pocket_diameter_mm?: number;
  magnet_pocket_depth_mm?: number;
  map_label?: string;
  map_label_text_height_mm?: number;
  keychain_mode?: boolean;
  keychain_label?: string;
  keychain_base_shape?: "rounded" | "capsule" | "tag" | "octagon" | "token" | "heart" | "house";
  keychain_label2?: string;
  keychain_label2_text_height_mm?: number;
  keychain_back_label?: string;
  keychain_back_text_height_mm?: number;
  keychain_back_engrave_mm?: number;
  keychain_layout_rotation_deg?: number;
  keychain_loop_style?: "round" | "teardrop" | "slot" | "side-tab";
  keychain_loop_angle_deg?: number;
  keychain_body_width_mm?: number;
  keychain_body_height_mm?: number;
  keychain_map_x_mm?: number;
  keychain_map_y_mm?: number;
  keychain_map_width_mm?: number;
  keychain_map_height_mm?: number;
  keychain_map_rotation_deg?: number;
  keychain_loop_center_x_mm?: number;
  keychain_loop_center_y_mm?: number;
  keychain_label_center_x_mm?: number;
  keychain_label_center_y_mm?: number;
  keychain_label_angle_deg?: number;
  keychain_loop_outer_radius_mm?: number;
  keychain_loop_inner_radius_mm?: number;
  keychain_corner_radius_mm?: number;
  keychain_label_band_height_mm?: number;
  keychain_label_raise_mm?: number;
  keychain_label_text_height_mm?: number;
  keychain_label_width_mm?: number;
  keychain_label_stroke_mm?: number;
  keychain_label_font_style?: "block" | "wide" | "condensed";
  keychain_rim_width_mm?: number;
  /** 4 corners of rotated rect [[lon, lat], ...] for precise OSM clipping */
  zone_polygon_coords?: Array<[number, number]>;
  keychain_rim_height_mm?: number;
  // Fast preview (~30s): skip Blender grooves + manifold cleanup, terrain 80x80
  preview_mode?: boolean;
  preview_include_base?: boolean;
  preview_include_roads?: boolean;
  preview_include_buildings?: boolean;
  preview_include_water?: boolean;
  preview_include_parks?: boolean;
}

export interface GenerationResponse {
  task_id: string;
  status: string;
}

export interface TaskStatus {
  task_id: string;
  status: string;
  progress: number;
  message: string;
  download_url: string | null;
  download_url_stl?: string | null;
  download_url_3mf?: string | null;
  firebase_url?: string | null;
  // QA / print-quality outcome from the backend gate (non-blocking)
  print_quality?: {
    status: "ok" | "warning" | "failed";
    warnings?: string[];
    report?: string | null;
  } | null;
  preview_3mf?: string | null;  // Основне прев'ю в 3MF форматі
  firebase_preview_3mf?: string | null;  // Firebase URL для основного прев'ю
  preview_parts?: {
    base?: string | null;
    roads?: string | null;
    buildings?: string | null;
    water?: string | null;
    parks?: string | null;
  };
  keychain_manifest?: {
    mode?: string;
    print_rules?: Record<string, any>;
    dimensions?: Record<string, number>;
    layers?: Record<string, {
      present?: boolean;
      vertices?: number;
      faces?: number;
      size_mm?: number[];
      z_min_mm?: number;
      z_max_mm?: number;
    }>;
  } | null;
  firebase_preview_parts?: {
    base?: string | null;
    roads?: string | null;
    buildings?: string | null;
    water?: string | null;
    parks?: string | null;
  };
}

export interface BatchTaskStatusResponse {
  task_id: string;
  status: "multiple";
  tasks: TaskStatus[];
  total: number;
  completed: number;
  all_task_ids: string[];
}

export type StatusResponse = TaskStatus | BatchTaskStatusResponse;

export const api = {
  async generateModel(request: GenerationRequest): Promise<GenerationResponse> {
    const response = await axios.post<GenerationResponse>(
      `${API_BASE_URL}/api/generate`,
      request
    );
    return response.data;
  },

  async getStatus(taskId: string): Promise<StatusResponse> {
    const response = await axios.get<StatusResponse>(
      `${API_BASE_URL}/api/status/${taskId}`
    );
    return response.data;
  },

  async downloadModel(
    taskId: string,
    format?: "stl" | "3mf" | "glb",
    part?: "base" | "roads" | "buildings" | "water" | "parks"
  ): Promise<Blob> {
    const params = new URLSearchParams();
    if (format) params.set("format", format);
    if (part) params.set("part", part);
    const qs = params.toString();
    const response = await axios.get(
      `${API_BASE_URL}/api/download/${taskId}${qs ? `?${qs}` : ""}`,
      {
        responseType: "blob",
        timeout: 600000, // 10 minutes
      }
    );
    return response.data;
  },

  async downloadFile(url: string): Promise<Blob> {
    const response = await axios.get(
      url.startsWith("http") ? url : `${API_BASE_URL}${url}`,
      {
        responseType: "blob",
        timeout: 600000, // 10 minutes (was 5)
      }
    );
    return response.data;
  },

  async generateHexagonalGrid(bounds: {
    north: number;
    south: number;
    east: number;
    west: number;
    hex_size_m?: number;
    grid_type?: "hexagonal" | "square" | "circle";
  }): Promise<{
    geojson: any;
    hex_count: number;
    is_valid: boolean;
    validation_errors: string[];
  }> {
    const response = await axios.post(
      `${API_BASE_URL}/api/hexagonal-grid`,
      {
        ...bounds,
        hex_size_m: bounds.hex_size_m || 300.0,
        grid_type: bounds.grid_type || "hexagonal",
      }
    );
    return response.data;
  },

  async generateZones(
    zones: any[],
    params: GenerationRequest
  ): Promise<GenerationResponse & { all_task_ids?: string[] }> {
    const response = await axios.post(
      `${API_BASE_URL}/api/generate-zones`,
      {
        zones,
        ...params,
      }
    );
    return response.data;
  },

  async getAccountModels(): Promise<AccountResponse> {
    const response = await axios.get<AccountResponse>(
      `${API_BASE_URL}/api/account/models`
    );
    return response.data;
  },

  async cancelTask(taskId: string): Promise<{ cancelled: boolean; count?: number }> {
    const response = await axios.delete<{ cancelled: boolean; count?: number }>(
      `${API_BASE_URL}/api/task/${taskId}`
    );
    return response.data;
  },

  async getAdminOrders(token?: string): Promise<{ orders: any[] }> {
    const params: Record<string, string> = {};
    if (token) params["token"] = token;
    const response = await axios.get<{ orders: any[] }>(
      `${API_BASE_URL}/api/admin/orders`,
      { params }
    );
    return response.data;
  },

  async startOrderGeneration(
    orderId: string,
    token?: string
  ): Promise<{ task_id: string; status: string; all_task_ids?: string[] }> {
    const params: Record<string, string> = {};
    if (token) params["token"] = token;
    const response = await axios.post<{ task_id: string; status: string }>(
      `${API_BASE_URL}/api/admin/orders/${orderId}/generate`,
      {},
      { params }
    );
    return response.data;
  },
};

