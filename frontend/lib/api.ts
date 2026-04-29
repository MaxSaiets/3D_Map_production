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
}

export interface GenerationResponse {
  task_id: string;
  status: string;
}

export interface FastPreviewRequest {
  north: number;
  south: number;
  east: number;
  west: number;
  polygon_geojson?: any;
  include_terrain?: boolean;
}

export interface FastPreviewResponse {
  preview_id: string;
  created_at: string;
  mode: "fast_preview";
  bounds: { north: number; south: number; east: number; west: number };
  center: { lat: number; lon: number };
  polygon?: any;
  layers: {
    buildings: any;
    roads: any;
    water: any;
    parks: any;
  };
  terrain: { enabled: boolean; mode: string };
  metrics: Record<string, number | string | boolean>;
}

export interface SiteOrderRequest {
  preview_id?: string;
  customer_name: string;
  contact: string;
  message?: string;
  city?: string;
  bounds: Record<string, any>;
  polygon_geojson?: any;
  selected_zones?: any[];
  options?: Record<string, any>;
}

export interface SiteOrderResponse {
  id: string;
  status: string;
  created_at: string;
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
  async generatePreview(request: FastPreviewRequest): Promise<FastPreviewResponse> {
    const response = await axios.post<FastPreviewResponse>(
      `${API_BASE_URL}/api/preview`,
      request,
      { timeout: 60000 }
    );
    return response.data;
  },

  async submitOrder(request: SiteOrderRequest): Promise<SiteOrderResponse> {
    const response = await axios.post<SiteOrderResponse>(
      `${API_BASE_URL}/api/orders`,
      request
    );
    return response.data;
  },

  async getAdminOrders(token?: string): Promise<{ orders: any[] }> {
    const response = await axios.get<{ orders: any[] }>(
      `${API_BASE_URL}/api/admin/orders`,
      token ? { headers: { "X-Admin-Token": token } } : undefined
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

