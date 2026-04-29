import axios from "axios";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "";

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
  is_ams_mode: boolean;
  preview_include_base?: boolean;
  preview_include_roads?: boolean;
  preview_include_buildings?: boolean;
  preview_include_water?: boolean;
  preview_include_parks?: boolean;
  context_padding_m?: number;
}

export interface PreviewRequest {
  north: number;
  south: number;
  east: number;
  west: number;
  polygon_geojson?: any;
  include_terrain?: boolean;
  include_roads?: boolean;
  include_buildings?: boolean;
  include_water?: boolean;
  include_parks?: boolean;
  road_width_multiplier?: number;
  building_min_height?: number;
  building_height_multiplier?: number;
  model_size_mm?: number;
  terrain_z_scale?: number;
  terrain_resolution?: number;
  road_height_mm?: number;
  road_embed_mm?: number;
  building_foundation_mm?: number;
  building_embed_mm?: number;
  water_depth?: number;
  parks_height_mm?: number;
  parks_embed_mm?: number;
  generation_request?: GenerationRequest;
}

export interface FastPreviewResponse {
  preview_id: string;
  preview_status?: "processing" | "ready" | "failed";
  cached: boolean;
  bounds: { north: number; south: number; east: number; west: number };
  center: { lat: number; lng: number };
  selection?: any;
  layers: {
    terrain: { enabled: boolean };
    roads: any;
    buildings: any;
    water: any;
    parks: any;
  };
  metrics: {
    buildings: number;
    roads: number;
    water: number;
    parks: number;
    elapsed_ms: number;
  };
  model_logic?: Record<string, any>;
}

export interface SiteOrderRequest {
  name: string;
  contact: string;
  city: string;
  bounds: { north: number; south: number; east: number; west: number };
  polygon_geojson?: any;
  preview_id?: string;
  model_size_mm: number;
  material: string;
  layers: Record<string, boolean>;
  price_uah?: number;
  comment?: string;
  area_mode?: string;
  selected_zones?: any[];
  grid_type?: string;
  hex_size_m?: number;
  preview_metrics?: Record<string, any>;
  model_logic?: Record<string, any>;
  generation_request?: GenerationRequest;
}

export interface GenerationResponse {
  task_id: string;
  status: string;
  message?: string;
  all_task_ids?: string[];
}

export interface AccountUsage {
  free_limit: number;
  used: number;
  completed: number;
  remaining: number;
}

export interface AccountModel {
  id: string;
  task_id: string;
  title: string;
  city: string;
  status: string;
  progress: number;
  message?: string;
  created_at: string;
  updated_at?: string;
  finished_at?: string;
  model_size_mm?: number;
  material?: string;
  layers?: Record<string, boolean>;
  bounds?: { north: number; south: number; east: number; west: number };
  preview_snapshot?: FastPreviewResponse | null;
  download_url?: string | null;
  download_url_3mf?: string | null;
  download_url_stl?: string | null;
  preview_3mf?: string | null;
  firebase_url?: string | null;
  error?: string | null;
}

export interface AccountResponse {
  profile: {
    uid: string;
    email?: string;
    name?: string;
    picture?: string;
    plan?: string;
  };
  usage: AccountUsage;
  recent_models?: AccountModel[];
  models?: AccountModel[];
}

export interface AccountGenerateRequest {
  title: string;
  city: string;
  preview_id?: string;
  preview_snapshot?: FastPreviewResponse | null;
  bounds: { north: number; south: number; east: number; west: number };
  polygon_geojson?: any;
  model_size_mm: number;
  material: string;
  layers: Record<string, boolean>;
  generation_request: GenerationRequest;
}

let authTokenProvider: (() => Promise<string | null>) | null = null;

export function setApiAuthTokenProvider(provider: (() => Promise<string | null>) | null) {
  authTokenProvider = provider;
}

async function authHeaders() {
  const token = authTokenProvider ? await authTokenProvider() : null;
  return token ? { Authorization: `Bearer ${token}` } : {};
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
  preview_3mf?: string | null;  // Основне прев'ю в 3MF форматі
  firebase_preview_3mf?: string | null;  // Firebase URL для основного прев'ю
  preview_parts?: {
    base?: string | null;
    roads?: string | null;
    buildings?: string | null;
    water?: string | null;
    parks?: string | null;
  };
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
  async createFastPreview(request: PreviewRequest): Promise<FastPreviewResponse> {
    const response = await axios.post<FastPreviewResponse>(`${API_BASE_URL}/api/preview`, request, {
      timeout: 45000,
    });
    return response.data;
  },

  async createSiteOrder(request: SiteOrderRequest): Promise<{ ok: boolean; order_id: string }> {
    const response = await axios.post(`${API_BASE_URL}/api/orders`, request, {
      timeout: 30000,
    });
    return response.data;
  },

  async getAdminOrders(token?: string): Promise<{ orders: any[] }> {
    const params = token ? `?token=${encodeURIComponent(token)}` : "";
    const response = await axios.get(`${API_BASE_URL}/api/admin/orders${params}`);
    return response.data;
  },

  async startOrderGeneration(orderId: string, token?: string): Promise<GenerationResponse> {
    const params = token ? `?token=${encodeURIComponent(token)}` : "";
    const response = await axios.post<GenerationResponse>(
      `${API_BASE_URL}/api/admin/orders/${encodeURIComponent(orderId)}/generate${params}`,
      {},
      { timeout: 30000 }
    );
    return response.data;
  },

  async generateModel(request: GenerationRequest): Promise<GenerationResponse> {
    const response = await axios.post<GenerationResponse>(
      `${API_BASE_URL}/api/generate`,
      request
    );
    return response.data;
  },

  async getAccount(): Promise<AccountResponse> {
    const response = await axios.get<AccountResponse>(`${API_BASE_URL}/api/account/me`, {
      headers: await authHeaders(),
    });
    return response.data;
  },

  async getAccountModels(): Promise<AccountResponse> {
    const response = await axios.get<AccountResponse>(`${API_BASE_URL}/api/account/models`, {
      headers: await authHeaders(),
    });
    return response.data;
  },

  async startAccountGeneration(request: AccountGenerateRequest): Promise<GenerationResponse> {
    const response = await axios.post<GenerationResponse>(
      `${API_BASE_URL}/api/account/models/generate`,
      request,
      {
        headers: await authHeaders(),
        timeout: 30000,
      }
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
    format?: "stl" | "3mf",
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
};

