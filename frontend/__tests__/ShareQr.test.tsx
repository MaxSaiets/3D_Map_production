/**
 * @jest-environment jsdom
 */
import { render, screen, waitFor } from "@testing-library/react";
import { ShareQr } from "@/components/ShareQr";

jest.mock("qrcode", () => ({
  toDataURL: jest.fn().mockResolvedValue("data:image/png;base64,AAAA"),
}));

describe("ShareQr", () => {
  it("renders the QR image and caption", async () => {
    render(
      <ShareQr url="https://monadruk.com/share/abc" label="Відкрити на телефоні" />
    );

    const img = await waitFor(() => screen.getByRole("img"));
    expect(img).toHaveAttribute("src", "data:image/png;base64,AAAA");
    expect(screen.getByText("Відкрити на телефоні")).toBeInTheDocument();
  });
});
