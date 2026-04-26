/**
 * @jest-environment jsdom
 */
import axios from "axios";
import { api } from "@/lib/api";

jest.mock("axios");
const mockedAxios = axios as jest.Mocked<typeof axios>;

describe("API Client", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe("generateModel", () => {
    it("calls generate endpoint with the current request contract", async () => {
      const mockResponse = {
        task_id: "test-task-id",
        status: "processing",
      };

      mockedAxios.post.mockResolvedValue({ data: mockResponse });

      const request = {
        north: 50.455,
        south: 50.45,
        east: 30.53,
        west: 30.52,
        road_width_multiplier: 1.0,
        road_height_mm: 0.5,
        road_embed_mm: 0.3,
        building_min_height: 2.0,
        building_height_multiplier: 1.0,
        building_foundation_mm: 0.5,
        building_embed_mm: 0.2,
        water_depth: 2.0,
        terrain_enabled: true,
        terrain_z_scale: 1.5,
        terrain_base_thickness_mm: 0.3,
        terrain_resolution: 180,
        terrarium_zoom: 15,
        export_format: "3mf" as const,
        model_size_mm: 100,
        is_ams_mode: false,
      };

      const result = await api.generateModel(request);

      expect(mockedAxios.post).toHaveBeenCalledWith(expect.stringContaining("/api/generate"), request);
      expect(result).toEqual(mockResponse);
    });
  });

  describe("getStatus", () => {
    it("calls status endpoint with task id", async () => {
      const mockResponse = {
        task_id: "test-task-id",
        status: "processing",
        progress: 50,
        message: "Обробка...",
        download_url: null,
      };

      mockedAxios.get.mockResolvedValue({ data: mockResponse });

      const result = await api.getStatus("test-task-id");

      expect(mockedAxios.get).toHaveBeenCalledWith(expect.stringContaining("/api/status/test-task-id"));
      expect(result).toEqual(mockResponse);
    });
  });

  describe("downloadModel", () => {
    it("downloads a model file", async () => {
      const mockBlob = new Blob(["test content"], { type: "application/octet-stream" });
      mockedAxios.get.mockResolvedValue({ data: mockBlob });

      const result = await api.downloadModel("test-task-id");

      expect(mockedAxios.get).toHaveBeenCalledWith(expect.stringContaining("/api/download/test-task-id"), {
        responseType: "blob",
        timeout: 600000,
      });
      expect(result).toBeInstanceOf(Blob);
    });

    it("downloads a model file with format query", async () => {
      const mockBlob = new Blob(["test content"], { type: "application/octet-stream" });
      mockedAxios.get.mockResolvedValue({ data: mockBlob });

      await api.downloadModel("test-task-id", "stl");

      expect(mockedAxios.get).toHaveBeenCalledWith(
        expect.stringContaining("/api/download/test-task-id?format=stl"),
        {
          responseType: "blob",
          timeout: 600000,
        },
      );
    });

    it("downloads a model file with format and part query", async () => {
      const mockBlob = new Blob(["test content"], { type: "application/octet-stream" });
      mockedAxios.get.mockResolvedValue({ data: mockBlob });

      await api.downloadModel("test-task-id", "stl", "roads");

      expect(mockedAxios.get).toHaveBeenCalledWith(
        expect.stringContaining("/api/download/test-task-id?format=stl&part=roads"),
        {
          responseType: "blob",
          timeout: 600000,
        },
      );
    });
  });
});
