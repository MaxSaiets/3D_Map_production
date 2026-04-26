/**
 * @jest-environment jsdom
 */
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { ControlPanel } from "@/components/ControlPanel";
import { api } from "@/lib/api";
import { useGenerationStore } from "@/store/generation-store";

jest.mock("@/store/generation-store");
jest.mock("@/lib/api");

const mockUseGenerationStore = useGenerationStore as jest.MockedFunction<typeof useGenerationStore>;
const mockApi = api as jest.Mocked<typeof api>;

describe("ControlPanel", () => {
  const mockStore = {
    selectedArea: null,
    isGenerating: false,
    taskGroupId: null,
    taskIds: [],
    activeTaskId: null,
    taskStatuses: {},
    showAllZones: false,
    progress: 0,
    status: "",
    downloadUrl: null,
    roadWidthMultiplier: 1.0,
    roadHeightMm: 0.5,
    roadEmbedMm: 0.3,
    buildingMinHeight: 2.0,
    buildingHeightMultiplier: 1.0,
    buildingFoundationMm: 0.5,
    buildingEmbedMm: 0.2,
    waterDepth: 2.0,
    terrainEnabled: true,
    terrainZScale: 1.5,
    terrainBaseThicknessMm: 0.3,
    terrainResolution: 180,
    terrariumZoom: 15,
    exportFormat: "3mf" as const,
    modelSizeMm: 100,
    isAmsMode: false,
    previewIncludeBase: true,
    previewIncludeRoads: true,
    previewIncludeBuildings: true,
    previewIncludeWater: true,
    previewIncludeParks: true,
    setRoadWidthMultiplier: jest.fn(),
    setRoadHeightMm: jest.fn(),
    setRoadEmbedMm: jest.fn(),
    setBuildingMinHeight: jest.fn(),
    setBuildingHeightMultiplier: jest.fn(),
    setBuildingFoundationMm: jest.fn(),
    setBuildingEmbedMm: jest.fn(),
    setWaterDepth: jest.fn(),
    setTerrainEnabled: jest.fn(),
    setTerrainZScale: jest.fn(),
    setTerrainBaseThicknessMm: jest.fn(),
    setTerrainResolution: jest.fn(),
    setTerrariumZoom: jest.fn(),
    setExportFormat: jest.fn(),
    setModelSizeMm: jest.fn(),
    setAmsMode: jest.fn(),
    setPreviewIncludeBase: jest.fn(),
    setPreviewIncludeRoads: jest.fn(),
    setPreviewIncludeBuildings: jest.fn(),
    setPreviewIncludeWater: jest.fn(),
    setPreviewIncludeParks: jest.fn(),
    setGenerating: jest.fn(),
    setTaskGroup: jest.fn(),
    setActiveTaskId: jest.fn(),
    setTaskStatuses: jest.fn(),
    setBatchZoneMetaByTaskId: jest.fn(),
    setShowAllZones: jest.fn(),
    updateProgress: jest.fn(),
    setDownloadUrl: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
    mockUseGenerationStore.mockReturnValue(mockStore as any);
  });

  it("renders redesigned essentials section", () => {
    render(<ControlPanel />);

    expect(screen.getByText("Керуйте потоком без зайвого шуму")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Згенерувати 3D модель/i })).toBeInTheDocument();
  });

  it("disables the primary action when no area is selected", () => {
    render(<ControlPanel />);

    expect(screen.getByRole("button", { name: /Згенерувати 3D модель/i })).toBeDisabled();
  });

  it("enables the primary action when an area is selected", () => {
    mockUseGenerationStore.mockReturnValue({
      ...mockStore,
      selectedArea: {} as any,
    } as any);

    render(<ControlPanel />);

    expect(screen.getByRole("button", { name: /Згенерувати 3D модель/i })).not.toBeDisabled();
  });

  it("calls generateModel and updates task group", async () => {
    const mockBounds = {
      getNorth: () => 50.455,
      getSouth: () => 50.45,
      getEast: () => 30.53,
      getWest: () => 30.52,
    };

    mockUseGenerationStore.mockReturnValue({
      ...mockStore,
      selectedArea: mockBounds as any,
    } as any);

    mockApi.generateModel.mockResolvedValue({
      task_id: "test-task-id",
      status: "processing",
    });

    render(<ControlPanel />);
    fireEvent.click(screen.getByRole("button", { name: /Згенерувати 3D модель/i }));

    await waitFor(() => {
      expect(mockApi.generateModel).toHaveBeenCalled();
      expect(mockStore.setGenerating).toHaveBeenCalledWith(true);
      expect(mockStore.setTaskGroup).toHaveBeenCalledWith("test-task-id", ["test-task-id"]);
      expect(mockStore.setActiveTaskId).toHaveBeenCalledWith("test-task-id");
    });
  });

  it("updates road width slider", () => {
    render(<ControlPanel />);

    fireEvent.click(screen.getAllByRole("button", { name: /Дороги/i })[0]);
    fireEvent.change(screen.getByLabelText(/Ширина доріг/i), { target: { value: "1.5" } });

    expect(mockStore.setRoadWidthMultiplier).toHaveBeenCalledWith(1.5);
  });

  it("shows progress state when generation is running", () => {
    mockUseGenerationStore.mockReturnValue({
      ...mockStore,
      isGenerating: true,
      progress: 50,
      status: "Обробка...",
    } as any);

    render(<ControlPanel />);

    expect(screen.getAllByText("Обробка...").length).toBeGreaterThan(0);
    expect(screen.getByText("50%")).toBeInTheDocument();
  });

  it("shows download button when file is ready", () => {
    mockUseGenerationStore.mockReturnValue({
      ...mockStore,
      downloadUrl: "/api/download/test-id",
      activeTaskId: "ready-task",
    } as any);

    render(<ControlPanel />);

    expect(screen.getByRole("button", { name: /Завантажити модель/i })).toBeInTheDocument();
  });
});
