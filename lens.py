import jax.numpy as jnp
from jax import jit, vmap
from litho.zernike import zerniken  # Ensure this function is compatible with JAX

class Lens:
    """Model lens with JAX for efficient computation."""
    def __init__(self, na=1.35, nLiquid=1.414, wavelength=193.0, defocus=0.0, maskxpitch=1000, maskypitch=1000):
        self.na = na
        self.nLiquid = nLiquid
        self.wavelength = wavelength
        self.defocus = defocus
        self.maskxpitch = maskxpitch
        self.maskypitch = maskypitch
        self.Zn = [9]
        self.Cn = [0.0]

    def update(self):
        self.detaf = self.wavelength / (self.maskxpitch * self.na)
        self.detag = self.wavelength / (self.maskypitch * self.na)
        self.fnum = int(jnp.ceil(2 / self.detaf))
        self.gnum = int(jnp.ceil(2 / self.detag))

    def calPupil(self, shiftx=0, shifty=0):
        fx = jnp.linspace(-self.fnum * self.detaf, self.fnum * self.detaf, 2 * self.fnum + 1)
        fy = jnp.linspace(-self.gnum * self.detag, self.gnum * self.detag, 2 * self.gnum + 1)
        FX, FY = jnp.meshgrid(fx - shiftx, fy - shifty, indexing="xy")

        R = jnp.sqrt(FX ** 2 + FY ** 2)
        TH = jnp.arctan2(FY, FX)
        H = jnp.where(R <= 1.0, 1.0, 0.0)
        R = jnp.where(R > 1.0, 0.0, R)

        W = jnp.zeros((2 * self.gnum + 1, 2 * self.fnum + 1), dtype=jnp.complex64)
        for ii in range(len(self.Zn)):
            W += zerniken(self.Zn[ii], R, TH) * self.Cn[ii]

        if self.na < 1:
            W += self.defocus / self.wavelength * (jnp.sqrt(1 - (self.na ** 2) * (R ** 2)) - 1)
        elif self.na >= 1:
            W += (self.na ** 2) / (2 * self.wavelength) * self.defocus * (R ** 2)
        
        self.fdata = H * jnp.exp(-1j * 2 * (jnp.pi) * W)

    def calPSF(self):
        normlize = 1  # self.detaf * self.detag, adjust if necessary
        self.data = jnp.fft.fftshift(jnp.fft.ifft2(jnp.fft.ifftshift(self.fdata))) * normlize

class LensList(Lens):
    """List of lens with JAX for efficient computation."""
    def __init__(self):
        super().__init__()
        self.focusList = [0.0]
        self.focusCoef = [1.0]
        self.fDataList = []
        self.sDataList = []

    def calculate(self):
        self.update()
        for ii in self.focusList:
            self.defocus = ii
            self.calPupil()
            self.fDataList.append(self.fdata)
            self.calPSF()
            self.sDataList.append(self.data)

if __name__ == "__main__":
    lens_list = LensList()
    lens_list.na = 0.85
    lens_list.focusList = [-50, 0, 50]
    lens_list.focusCoef = [0.5, 1, 0.5]
    lens_list.calculate()

    lens = Lens()
    lens.na = 0.85
    lens.update()
    lens.calPupil()
    lens.calPSF()
